// Private submodules
mod shadergen;
mod gpuutils;

#[cfg(test)]
mod tests;

use std::result;
use std::iter;
use std::mem;
use std::rc::Rc;

use futures_intrusive::channel::shared::{Sender, channel};
use log::Level::Trace;
use log::{trace, debug, log_enabled};
use ndarray::{Array2, ArrayView2, Axis, s};
use rustc_hash::{FxHashSet, FxHashMap};
use serde::{Serialize, Deserialize};
use wgpu::{Buffer, Device};
use wgpu::util::DeviceExt;

use crate::error::*;
use crate::energy::*;
use gpuutils::{GPUCommon, bgl_entry, buffer_fits};
use shadergen::{ShaderPreproc, make_replacements,
    build_attack_update, build_attack_minimize, build_defend_update, build_defend_iterative};

// Spawn 64 threads with each workgroup invocation
const WORKGROUP_SIZE: u32 = 64;
// Initial size of buffers in u32's, if there is no data yet.
// Buffers with size 0 are not allowed.
const INITIAL_CAPACITY: u64 = 64;

#[derive(Serialize, Deserialize)]
struct SerdeGameGraph {
    conf: EnergyConf,
    adj: Vec<Vec<u32>>,
    weights: Vec<Vec<Vec<i32>>>,
    attacker_pos: Vec<bool>,
}

impl From<GameGraph> for SerdeGameGraph {
    fn from(graph: GameGraph) -> Self {
        let mut adj = Vec::new();
        let mut weights = Vec::new();
        for i in 0..graph.n_vertices() {
            adj.push(graph.adj(i).to_vec());
            let weight = graph.weights.array.slice(
                s![graph.row_offsets[i as usize] as usize .. graph.row_offsets[i as usize + 1] as usize, ..]);
            weights.push((&UpdateArray::from_array(weight.to_owned(), graph.conf)).into());
        }
        Self {
            conf: graph.conf,
            adj,
            weights,
            attacker_pos: graph.attacker_pos,
        }
    }
}

impl TryFrom<SerdeGameGraph> for GameGraph {
    type Error = &'static str;
    fn try_from(deserialized: SerdeGameGraph) -> std::result::Result<Self, Self::Error> {
        let n_vertices = deserialized.adj.len();
        if deserialized.weights.len() != n_vertices {
            return Err("Weight list size doesn't match adjacecy list size");
        }
        if deserialized.attacker_pos.len() != n_vertices {
            return Err("attacker_pos list has wrong length");
        }
        for (successors, weights) in deserialized.adj.iter().zip(&deserialized.weights) {
            if successors.len() != weights.len() {
                return Err("Out-degree in adjacency list doesn't match out-degree in weight list");
            }
            if !successors.iter().all(|&s| (s as usize) < n_vertices) {
                return Err("Node index too high");
            }
        }

        let flat_weights: Vec<_> = deserialized.weights.into_iter().flatten().collect();
        let weights: std::result::Result<UpdateArray, Self::Error> =
            UpdateArray::from_conf(flat_weights.as_slice(), deserialized.conf);
        let row_offsets = iter::once(0).chain(
            deserialized.adj.iter()
            .scan(0, |state, adj| {
                *state += adj.len() as u32;
                Some(*state)
            }))
            .collect();
        let column_indices = deserialized.adj.into_iter()
            .flatten()
            .collect();
        let mut graph = GameGraph {
            column_indices,
            row_offsets,
            weights: weights?,
            rev_column_indices: Vec::new(),
            rev_row_offsets: Vec::new(),
            attacker_pos: deserialized.attacker_pos,
            conf: deserialized.conf,
        };
        graph.make_reverse();
        Ok(graph)
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "SerdeGameGraph", into = "SerdeGameGraph")]
pub struct GameGraph {
    // CSR graph representation. colmumn_indices is all adjacency lists, flattened out
    pub column_indices: Vec<u32>,
    // row_offsets shows where each node's successors start in column_indices array
    pub row_offsets: Vec<u32>,
    // weights is structured like column_indices, they must have the same length
    pub weights: UpdateArray,

    // Reverse edges of graph, also as CSR
    pub rev_column_indices: Vec<u32>,
    pub rev_row_offsets: Vec<u32>,

    // Whether each node is an attacker position or not
    pub attacker_pos: Vec<bool>,
    conf: EnergyConf,
}

impl GameGraph {
    pub fn new<T>(
        n_vertices: u32,
        edges: Vec<(u32, u32, T)>,
        attacker_pos: Vec<bool>,
        conf: EnergyConf,
    ) -> Self
    where
        // Accept any type T for weights that can be turned into a row of an UpdateArray
        for<'a> UpdateArray: FromEnergyConf<&'a [T]>,
        T: Clone,
    {
        let mut adj = vec![vec![]; n_vertices as usize];
        let mut reverse = vec![vec![]; n_vertices as usize];
        let mut raw_weights = vec![vec![]; n_vertices as usize];
        for (from, to, e) in edges {
            adj[from as usize].push(to);
            reverse[to as usize].push(from);
            raw_weights[from as usize].push(e);
        }
        // Cumulative sum of all list lengths. 0 is prepended manually
        let row_offsets = iter::once(0).chain(
            adj.iter()
            .scan(0, |state, adj| {
                *state += adj.len() as u32;
                Some(*state)
            }))
            .collect();
        let column_indices = adj.into_iter()
            .flatten()
            .collect();
        let flat_weights: Vec<T> = raw_weights.into_iter()
            .flatten()
            .collect();
        let weights = UpdateArray::from_conf(&flat_weights, conf).unwrap();

        let rev_row_offsets = iter::once(0).chain(
            reverse.iter()
            .scan(0, |state, adj| {
                *state += adj.len() as u32;
                Some(*state)
            }))
            .collect();
        let rev_column_indices = reverse.into_iter().flatten().collect();

        Self {
            column_indices,
            row_offsets,
            weights,
            rev_column_indices,
            rev_row_offsets,
            attacker_pos,
            conf,
        }
    }

    pub fn empty(conf: EnergyConf) -> Self {
        Self {
            column_indices: Vec::new(),
            row_offsets: Vec::new(),
            weights: UpdateArray::empty(conf),
            rev_column_indices: Vec::new(),
            rev_row_offsets: Vec::new(),
            attacker_pos: Vec::new(),
            conf,
        }
    }

    #[inline]
    pub fn n_vertices(&self) -> u32 {
        self.attacker_pos.len() as u32
    }

    #[inline]
    pub fn get_conf(&self) -> EnergyConf {
        self.conf
    }

    #[inline]
    pub fn adj(&self, v: u32) -> &[u32] {
        let v = v as usize;
        &self.column_indices[self.row_offsets[v] as usize .. self.row_offsets[v + 1] as usize]
    }

    #[inline]
    pub fn reverse(&self, v: u32) -> &[u32] {
        let v = v as usize;
        &self.rev_column_indices[self.rev_row_offsets[v] as usize .. self.rev_row_offsets[v + 1] as usize]
    }

    pub fn make_reverse(&mut self) {
        let mut reverse = vec![vec![]; self.n_vertices() as usize];
        for from in 0..self.n_vertices() {
            for &to in self.adj(from) {
                reverse[to as usize].push(from);
            }
        }
        self.rev_row_offsets = iter::once(0).chain(
            reverse.iter()
            .scan(0, |state, adj| {
                *state += adj.len() as u32;
                Some(*state)
            }))
            .collect();
        self.rev_column_indices = reverse.into_iter().flatten().collect();
    }


    fn buffers(&self, device: &Device) -> (Buffer, Buffer) {
        let row_offsets_buffer = if self.row_offsets.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Input graph row offsets buffer (empty)"),
                size: 8,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input graph row offsets storage buffer"),
                contents: bytemuck::cast_slice(&self.row_offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };

        let weights_buffer = if self.weights.n_updates() == 0 {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Input graph edge weights buffer (empty)"),
                size: 8,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input graph edge weights storage buffer"),
                contents: self.weights.data(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };

        (row_offsets_buffer, weights_buffer)
    }

    fn bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Input graph bind group layout"),
            entries: &[
                bgl_entry(0, true), // graph row offsets
                bgl_entry(1, true), // graph edge weights
            ],
        })
    }

    fn bind_group_from_parts(
        buffers: &(Buffer, Buffer),
        layout: &wgpu::BindGroupLayout,
        device: &Device,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Input graph bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { // graph row offsets
                    binding: 0,
                    resource: buffers.0.as_entire_binding(),
                },
                wgpu::BindGroupEntry { // graph edge weights
                    binding: 1,
                    resource: buffers.1.as_entire_binding(),
                },
            ],
        })
    }

    fn bind_group(&self, device: &Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        let layout = Self::bind_group_layout(device);
        let buffers = self.buffers(device);
        let bind_group = Self::bind_group_from_parts(&buffers, &layout, device);
        (layout, bind_group)
    }
}


#[derive(Debug, Clone)]
pub struct EnergyGame {
    pub graph: GameGraph,
    // One array of energies for each node
    pub energies: Vec<EnergyArray>,
    pub to_reach: Vec<u32>,
}

impl EnergyGame {

    // Automatically set nodes to be reached to all defense nodes without outgoing edges
    pub fn standard_reach(graph: GameGraph) -> Self {
        let to_reach = (0..graph.n_vertices())
            .filter(|&v| !graph.attacker_pos[v as usize] && graph.adj(v).is_empty())
            .collect();
        Self::with_reach(graph, to_reach)
    }

    pub fn with_reach(graph: GameGraph, to_reach: Vec<u32>) -> Self {
        let mut energies = vec![EnergyArray::zero(0, graph.get_conf()); graph.n_vertices() as usize];
        for v in &to_reach {
            energies[*v as usize] = EnergyArray::zero(1, graph.get_conf());
        }
        EnergyGame {
            graph,
            energies,
            to_reach,
        }
    }

    pub async fn get_gpu_runner(&mut self) -> Result<GPURunner> {
        GPURunner::with_game(self).await
    }

    pub async fn run(&mut self) -> Result<&[EnergyArray]> {
        let mut runner = self.get_gpu_runner().await?;
        runner.execute_gpu().await
            // Return final energy table
            .map(|_| self.energies.as_slice())
    }
}


#[repr(C)]  // Needed for safely implementing Pod
#[derive(Clone, Copy, Debug)]
struct NodeOffsetAtk {
    node: u32,
    // Position in thet successor offsets array where the offsets of this node's successors start
    successor_offsets_idx: u32,
    // Position in the energy array where the energies for this node start
    offset: u32,
}

// Enable bytemucking for filling buffers
unsafe impl bytemuck::Zeroable for NodeOffsetAtk {}
unsafe impl bytemuck::Pod for NodeOffsetAtk {}


// More data is needed for defence shaders to also include offsets in the list of combinations
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NodeOffsetDef {
    node: u32,
    successor_offsets_idx: u32,
    energy_offset: u32,
    // Position in the sup array where the first of this node's suprema should be written to
    sup_offset: u32,
}

unsafe impl bytemuck::Zeroable for NodeOffsetDef {}
unsafe impl bytemuck::Pod for NodeOffsetDef {}


trait PlayerShader {
    fn name() -> &'static str;

    // Construct buffer for energies
    fn get_energies_buf(device: &Device, energies: &EnergyArray) -> Buffer {
        if energies.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Energies storage buffer empty", Self::name())),
                size: INITIAL_CAPACITY * mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Energies storage buffer", Self::name())),
                contents: energies.data(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            })
        }
    }

    fn get_successor_offsets_buf(device: &Device, successor_offsets: &[u32]) -> Buffer {
        if successor_offsets.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Successor offsets storage buffer empty", Self::name())),
                size: INITIAL_CAPACITY * mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Successor offsets storage buffer", Self::name())),
                contents: bytemuck::cast_slice(successor_offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        }
    }

    fn new_output_buf(device: &Device, size: u64, label: Option<&str>) -> (Buffer, Buffer) {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} {} buffer", Self::name(), label.unwrap_or_default())),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // For reading minima_buf on the CPU
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} {} staging buffer", Self::name(), label.unwrap_or_default())),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (buf, staging_buf)
    }
}


struct DefendIterShader {
    gpu: Rc<GPUCommon>,
    // Visit list items are (node, mem), where mem is the amount of requested memory for suprema.
    visit_list: FxHashMap<u32, u32>,

    energies: EnergyArray,
    energies_buf: Buffer,
    node_offsets: Vec<NodeOffsetDef>,
    node_offsets_buf: Buffer,
    successor_offsets_buf: Buffer,
    sup_buf: Buffer,
    sup_staging_buf: Buffer,
    status_buf: Buffer,
    status_staging_buf: Buffer,
    work_size_buf: Buffer,

    input_bind_group: wgpu::BindGroup,
    input_bind_group_layout: wgpu::BindGroupLayout,
    update_pipeline: wgpu::ComputePipeline,

    suprema_bind_group: wgpu::BindGroup,
    suprema_bind_group_layout: wgpu::BindGroupLayout,
    combine_pipeline: wgpu::ComputePipeline,
}

impl PlayerShader for DefendIterShader {
    fn name() -> &'static str { "Defend Iterative" }
}

impl DefendIterShader {
    const DEFAULT_SUPREMA_MEMORY: u32 = 64;

    fn new(
        gpu: Rc<GPUCommon>,
        conf: EnergyConf,
        graph_bind_group_layout: &wgpu::BindGroupLayout,
        preprocessor: &ShaderPreproc,
    ) -> DefendIterShader {
        let node_offsets_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Defend Iterative Node offsets storage buffer initial"),
            size: INITIAL_CAPACITY * mem::size_of::<NodeOffsetDef>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let energies_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Defend Iterative Energies storage buffer initial"),
            size: INITIAL_CAPACITY * u64::from(conf.energy_size()) * mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let successor_offsets_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Defend Iterative Successor offsets storage buffer initial"),
            size: INITIAL_CAPACITY * mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sup_bytes = INITIAL_CAPACITY * u64::from(conf.energy_size()) * mem::size_of::<u32>() as u64;
        let (sup_buf, sup_staging_buf) = Self::new_output_buf(
            &gpu.device, sup_bytes, Some("Suprema"));

        let (status_buf, status_staging_buf) = Self::new_output_buf(
            &gpu.device,
            INITIAL_CAPACITY * mem::size_of::<i32>() as u64,
            Some("Intersection status"),
        );

        let work_size_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Defend Iterative work size uniform buffer"),
            size: mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let input_bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Defend Iterative common input bind group layout"),
            entries: &[
                bgl_entry(0, false), // energies, writable
                bgl_entry(1, true),  // node offsets
                bgl_entry(2, true),  // successor offsets
                // work_size, uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let input_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} common input bind group (initial)", Self::name())),
            layout: &input_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: energies_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: node_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: successor_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: work_size_buf.as_entire_binding(),
                },
            ],
        });

        // A bind group can have at most 4 storage buffers, split across 2 bind groups.
        let suprema_bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Defend Iterative suprema of combinations shader bind group layout 1"),
            entries: &[
                bgl_entry(0, false), // suprema, writable
                bgl_entry(1, false), // status, writable
            ],
        });
        let suprema_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Defend Iterative combinations shader bind group (inital)"),
            layout: &suprema_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sup_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: status_buf.as_entire_binding(),
                },
            ],
        });

        let update_pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Defend Iterative suprema of combinations pipeline layout"),
            bind_group_layouts: &[&input_bind_group_layout, &graph_bind_group_layout],
            push_constant_ranges: &[],
        });
        let update_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Defend Iterative energy update shader module"),
            source: build_defend_update(preprocessor),
        });
        let update_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Defend Iterative energy update pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_shader,
            entry_point: "main",
        });

        let combine_pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Defend Iterative suprema of combinations pipeline layout"),
            bind_group_layouts: &[&input_bind_group_layout, &suprema_bind_group_layout],
            push_constant_ranges: &[],
        });
        let combine_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Defend Iterative suprema of combinations shader module"),
            source: build_defend_iterative(preprocessor),
        });
        let combine_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Defend Iterative suprema of combinations pipeline"),
            layout: Some(&combine_pipeline_layout),
            module: &combine_shader,
            entry_point: "intersection",
        });

        DefendIterShader {
            gpu,
            visit_list: FxHashMap::default(),
            energies: EnergyArray::empty(conf),
            energies_buf,
            node_offsets: Vec::new(),
            node_offsets_buf,
            successor_offsets_buf,
            sup_buf,
            sup_staging_buf,
            status_buf,
            status_staging_buf,
            work_size_buf,

            input_bind_group,
            input_bind_group_layout,
            update_pipeline,

            suprema_bind_group,
            suprema_bind_group_layout,
            combine_pipeline,
        }
    }

    #[inline]
    fn energies_size(&self) -> u64 {
        (self.energies.view().len() * mem::size_of::<u32>()) as u64
    }
    #[inline]
    fn node_offsets_size(&self) -> u64 {
        (self.node_offsets.len() * mem::size_of::<NodeOffsetDef>()) as u64
    }
    #[inline]
    fn sup_size(&self) -> u64 {
        let n_sup: u64 = self.node_offsets.last().map(|n| n.sup_offset.into()).unwrap_or_default();
        n_sup * u64::from(self.energies.get_conf().energy_size()) * mem::size_of::<u32>() as u64
    }
    #[inline]
    fn status_size(&self) -> u64 {
        ((self.node_offsets.len() - 1) * mem::size_of::<i32>()) as u64
    }

    fn update(&mut self,
        node_offsets: Vec<NodeOffsetDef>,
        successor_offsets: Vec<u32>,
        energies: EnergyArray,
    ) {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;

        queue.write_buffer(&self.work_size_buf, 0, bytemuck::bytes_of(&(node_offsets.len() as u32)));
        self.energies = energies;
        if self.energies_size() > self.energies_buf.size() {
            self.energies_buf = Self::get_energies_buf(device, &self.energies);
        } else {
            queue.write_buffer(&self.energies_buf, 0, self.energies.data());
        }

        self.node_offsets = node_offsets;
        if self.node_offsets_size() > self.node_offsets_buf.size() {
            self.node_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Defend Iterative node offsets storage buffer"),
                contents: bytemuck::cast_slice(&self.node_offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&self.node_offsets_buf, 0, bytemuck::cast_slice(&self.node_offsets));
        }

        if !buffer_fits(&successor_offsets, &self.successor_offsets_buf) {
            self.successor_offsets_buf = Self::get_successor_offsets_buf(device, &successor_offsets);
        } else {
            queue.write_buffer(&self.successor_offsets_buf, 0, bytemuck::cast_slice(&successor_offsets));
        }

        let sup_size = self.sup_size();
        if sup_size > self.sup_buf.size() {
            (self.sup_buf, self.sup_staging_buf) = Self::new_output_buf(
                device, sup_size, Some("Suprema"));
        }

        let status_size = self.status_size();
        if status_size > self.status_buf.size() {
            (self.status_buf, self.status_staging_buf) = Self::new_output_buf(
                device, status_size, Some("Intersection status"));
        }

        self.input_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Defend Iterative common input bind group (updated)"),
            layout: &self.input_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.energies_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.node_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.successor_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.work_size_buf.as_entire_binding(),
                },
            ],
        });

        self.suprema_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Defend Iterative combinations shader bind group (updated)"),
            layout: &self.suprema_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sup_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.status_buf.as_entire_binding(),
                },
            ],
        });
    }

    fn collect_data(
        &mut self,
        game: &EnergyGame,
    ) -> (Vec<NodeOffsetDef>, Vec<u32>, EnergyArray) {
        let mut energies = Array2::zeros((0, game.graph.get_conf().energy_size() as usize));
        let mut successor_offsets = Vec::new();
        let mut node_offsets = Vec::new();
        let mut sup_count = 0;

        let max_size = self.gpu.device.limits().max_storage_buffer_binding_size as usize - 256;
        let max_wg = self.gpu.device.limits().max_compute_workgroups_per_dimension;
        let visit_list: Vec<(u32, u32)> = self.visit_list.iter().map(|(&k, &v)| (k, v)).collect();
        for (node, mem) in visit_list {
            self.visit_list.remove(&node);

            // If any successor has no energies associated yet, skip this node
            if game.graph.adj(node).iter()
                .any(|&suc| game.energies[suc as usize].is_empty())
            { continue }

            node_offsets.push(NodeOffsetDef {
                node,
                successor_offsets_idx: successor_offsets.len() as u32,
                energy_offset: energies.nrows() as u32,
                sup_offset: sup_count,
            });
            for &successor in game.graph.adj(node) {
                successor_offsets.push(energies.nrows() as u32);
                energies.append(Axis(0), game.energies[successor as usize].view()).unwrap();
            }
            // Allocate the requested amount of suprema memory for this node
            sup_count = sup_count.saturating_add(mem);

            // Check limits
            let node_offsets_size = mem::size_of_val(&node_offsets);
            let successor_offsets_size = mem::size_of_val(&successor_offsets);
            let energies_size = energies.len() * mem::size_of::<u32>();
            let workgroup_count = (energies.nrows() as u32).div_ceil(WORKGROUP_SIZE);
            let sup_size = (sup_count * game.graph.get_conf().energy_size()) as usize * mem::size_of::<u32>();
            if workgroup_count > max_wg || // Update workgroups
                node_offsets.len() as u32 > max_wg || // Minimize workgroups
                sup_size > max_size ||
                energies_size > max_size ||
                node_offsets_size > max_size ||
                successor_offsets_size > max_size
            {
                self.visit_list.insert(node, mem);
                // This node doesn't fit, mark last node offset as final cap
                node_offsets.last_mut().unwrap().node = u32::MAX;
                break;
            }
        }

        if self.visit_list.is_empty() {
            node_offsets.push(NodeOffsetDef {
                // Last offset does not correspond to another starting node, mark with u32::MAX
                node: u32::MAX,
                successor_offsets_idx: successor_offsets.len() as u32,
                energy_offset: energies.nrows() as u32,
                sup_offset: sup_count,
            });
            successor_offsets.push(energies.nrows() as u32);
        }
        let earray = EnergyArray::from_array(energies, game.graph.get_conf());
        trace!("Defend Iterative data: {:?}\n{:?}\n{}", node_offsets, successor_offsets, earray);

        (node_offsets, successor_offsets, earray)
    }

    #[inline]
    fn prepare_run(&mut self, game: &EnergyGame) {
        let data = self.collect_data(game);
        self.update(data.0, data.1, data.2);
    }

    fn compute_pass(&self, encoder: &mut wgpu::CommandEncoder, graph_bind_group: &wgpu::BindGroup) {
        { // Compute pass for updating energies
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Defend Iterative compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.update_pipeline);
            cpass.set_bind_group(0, &self.input_bind_group, &[]);
            cpass.set_bind_group(1, graph_bind_group, &[]);

            let n_energies = self.node_offsets.last().map(|n| n.energy_offset).unwrap_or_default();
            let update_workgroup_count = n_energies.div_ceil(WORKGROUP_SIZE);
            cpass.dispatch_workgroups(update_workgroup_count, 1, 1);

            // Minima of suprema shader
            cpass.set_pipeline(&self.combine_pipeline);
            cpass.set_bind_group(1, &self.suprema_bind_group, &[]);
            // One workgroup for each node
            cpass.dispatch_workgroups(self.node_offsets.len().max(1) as u32 - 1, 1, 1);
        }
        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.status_buf, 0, &self.status_staging_buf, 0, self.status_size());
        encoder.copy_buffer_to_buffer(
            &self.sup_buf, 0, &self.sup_staging_buf, 0, self.sup_size());
    }

    const MAPPED_BUFS: usize = 2;

    // Returns the number of buffers that were actually mapped
    fn map_buffers(&self, sender: &Sender<result::Result<(), wgpu::BufferAsyncError>>) -> usize {
        if self.node_offsets.len() <= 1 {
            return 0;
        }
        let status_buffer_slice = self.status_staging_buf.slice(.. self.status_size());
        let sup_buffer_slice = self.sup_staging_buf.slice(.. self.sup_size());
        let sender0 = sender.clone();
        status_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender0.try_send(v).expect("Channel should be writable");
        });
        let sender1 = sender.clone();
        sup_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender1.try_send(v).expect("Channel should be writable");
        });
        Self::MAPPED_BUFS
    }

    fn process_results(&mut self, game: &mut EnergyGame) -> Vec<u32> {
        if self.node_offsets.len() <= 1 {
            return Vec::new();
        }
        let status_data = self.status_staging_buf.slice(.. self.status_size()).get_mapped_range();
        let status: &[i32] = bytemuck::cast_slice(&status_data);

        let sup_data = self.sup_staging_buf.slice(.. self.sup_size()).get_mapped_range();
        let sup_slice: &[u32] = bytemuck::cast_slice(&sup_data);
        let energy_size = game.graph.get_conf().energy_size() as usize;
        let n_sup = sup_slice.len() / energy_size;
        let sup_array = ArrayView2::from_shape((n_sup, energy_size), sup_slice)
            .expect("Suprema array has invalid shape");

        if log_enabled!(Trace) {
            trace!("Defend Iterative Status values:\n{:?}", status);
            let suprema = EnergyArray::from_array(sup_array.to_owned(), game.graph.get_conf());
            trace!("Defend Iterative Suprema:\n{}", suprema);
        }

        let mut changed_nodes = Vec::new();
        let last = self.node_offsets.len().max(1) - 1;
        for (node, &status) in self.node_offsets[..last].iter().zip(status) {
            if status < 0 {
                // A negative status means calculation was aborted due to insufficient memory
                let next_memsize = status.unsigned_abs().next_power_of_two().max(128);
                self.visit_list.insert(node.node, next_memsize);
                continue;
            }
            let start = node.sup_offset as usize;
            let end = start + status as usize;
            let new_array = sup_array.slice(s![start..end, ..]);

            if new_array != game.energies[node.node as usize] {
                // Copy data only after ensuring it is new
                let new_energies = EnergyArray::from_array(new_array.to_owned(), game.graph.get_conf());
                // Write new energies
                game.energies[node.node as usize] = new_energies;
                changed_nodes.push(node.node);
            }
        }

        // Unmap buffers
        drop(status_data);
        self.status_staging_buf.unmap();
        drop(sup_data);
        self.sup_staging_buf.unmap();
        changed_nodes
    }
}


// Type used to store bit-packed minima flags for CPU.
type MinFlags = u64;

struct AttackShader {
    gpu: Rc<GPUCommon>,
    visit_list: FxHashSet<u32>,
    energies: EnergyArray,
    energies_buf: Buffer,
    energies_staging_buf: Buffer,
    node_offsets: Vec<NodeOffsetAtk>,
    node_offsets_buf: Buffer,
    successor_offsets_buf: Buffer,
    minima_buf: Buffer,
    minima_staging_buf: Buffer,
    changed_buf: Buffer,
    changed_staging_buf: Buffer,
    work_size_buf: Buffer,

    input_bind_group: wgpu::BindGroup,
    input_bind_group_layout: wgpu::BindGroupLayout,
    update_pipeline: wgpu::ComputePipeline,

    minima_bind_group: wgpu::BindGroup,
    minima_bind_group_layout: wgpu::BindGroupLayout,
    minima_pipeline: wgpu::ComputePipeline,
}

impl PlayerShader for AttackShader {
    fn name() -> &'static str { "Attack" }
}

impl AttackShader {
    fn new(
        gpu: Rc<GPUCommon>,
        conf: EnergyConf,
        graph_bind_group_layout: &wgpu::BindGroupLayout,
        preprocessor: &ShaderPreproc,
    ) -> AttackShader {
        let node_offsets_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attack Node offsets buffer initial"),
            size: INITIAL_CAPACITY * mem::size_of::<NodeOffsetAtk>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let energies_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attack Energies buffer initial"),
            size: INITIAL_CAPACITY * u64::from(conf.energy_size()) * mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let energies_staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attack Energies staging buffer initial"),
            size: energies_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let successor_offsets_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attack Successor offsets buffer initial"),
            size: INITIAL_CAPACITY * mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (minima_buf, minima_staging_buf) = Self::new_output_buf(
            &gpu.device, INITIAL_CAPACITY, Some("Minima flags"));

        let (changed_buf, changed_staging_buf) = Self::new_output_buf(
            &gpu.device, INITIAL_CAPACITY, Some("Energies changed flags"));

        let work_size_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attack work size uniform buffer"),
            size: mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let input_bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Attack input bind group layout"),
            entries: &[
                bgl_entry(0, false), // energies, writable
                bgl_entry(1, true),  // node offsets
                bgl_entry(2, true),  // successor offsets
                bgl_entry(3, false), // changed, writable
                // work_size, uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let input_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Attack input bind group"),
            layout: &input_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: energies_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: node_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: successor_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: changed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: work_size_buf.as_entire_binding(),
                },
            ],
        });

        let minima_bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Attack minima bind group layout"),
            entries: &[
                bgl_entry(0, false), // minima, writable
            ],
        });

        let minima_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Attack minima bind group"),
            layout: &minima_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: minima_buf.as_entire_binding(),
                },
            ],
        });

        let update_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Attack update shader module"),
            source: build_attack_update(preprocessor),
        });
        let update_pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Attack update pipeline layout"),
            bind_group_layouts: &[&input_bind_group_layout, &graph_bind_group_layout],
            push_constant_ranges: &[],
        });
        let update_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Attack update compute pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_shader,
            entry_point: "update",
        });

        let minima_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Attack minimize shader module"),
            source: build_attack_minimize(preprocessor),
        });
        let minima_pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Attack minimize pipeline layout"),
            bind_group_layouts: &[&input_bind_group_layout, &minima_bind_group_layout],
            push_constant_ranges: &[],
        });
        let minima_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Attack minima pipeline"),
            layout: Some(&minima_pipeline_layout),
            module: &minima_shader,
            entry_point: "minimize",
        });

        AttackShader {
            gpu,
            visit_list: FxHashSet::default(),
            energies: EnergyArray::empty(conf),
            energies_buf,
            energies_staging_buf,
            node_offsets: Vec::new(),
            node_offsets_buf,
            successor_offsets_buf,
            minima_buf,
            minima_staging_buf,
            changed_buf,
            changed_staging_buf,
            work_size_buf,

            input_bind_group,
            input_bind_group_layout,
            update_pipeline,

            minima_bind_group,
            minima_bind_group_layout,
            minima_pipeline,
        }
    }

    #[inline]
    fn energies_size(&self) -> u64 {
        (self.energies.view().len() * mem::size_of::<u32>()) as u64
    }
    #[inline]
    fn node_offsets_size(&self) -> u64 {
        (self.node_offsets.len() * mem::size_of::<NodeOffsetAtk>()) as u64
    }
    #[inline]
    fn minima_size(&self) -> u64 {
        const BITS: i64 = MinFlags::BITS as i64;
        let n = self.node_offsets.last().map(|n| n.offset).unwrap_or_default();
        (((n as i64 - 1) / BITS + 1) * BITS / 8) as u64
    }
    #[inline]
    fn changed_size(&self) -> u64 {
        ((self.node_offsets.len() - 1) * mem::size_of::<u32>()) as u64
    }

    fn update(&mut self,
        node_offsets: Vec<NodeOffsetAtk>,
        successor_offsets: Vec<u32>,
        energies: EnergyArray,
    ) {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;

        queue.write_buffer(&self.work_size_buf, 0, bytemuck::bytes_of(&(node_offsets.len() as u32)));
        self.energies = energies;
        if self.energies_size() > self.energies_buf.size() {
            self.energies_buf = Self::get_energies_buf(device, &self.energies);
            self.energies_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Attack Energies staging buffer"),
                size: self.energies_buf.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        } else {
            queue.write_buffer(&self.energies_buf, 0, self.energies.data());
        }

        self.node_offsets = node_offsets;
        if self.node_offsets_size() > self.node_offsets_buf.size() {
            self.node_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Node offsets storage buffer", Self::name())),
                contents: bytemuck::cast_slice(&self.node_offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&self.node_offsets_buf, 0, bytemuck::cast_slice(&self.node_offsets));
        }

        if !buffer_fits(&successor_offsets, &self.successor_offsets_buf) {
            self.successor_offsets_buf = Self::get_successor_offsets_buf(device, &successor_offsets);
        } else {
            queue.write_buffer(&self.successor_offsets_buf, 0, bytemuck::cast_slice(&successor_offsets));
        }

        let minima_size = self.minima_size();
        if minima_size > self.minima_buf.size() {
            (self.minima_buf, self.minima_staging_buf) = Self::new_output_buf(
                device, minima_size, Some("Minima flags"));
        }

        let changed_size = self.changed_size();
        if changed_size > self.changed_buf.size() {
            (self.changed_buf, self.changed_staging_buf) = Self::new_output_buf(
                device, changed_size, Some("Energies changed flags"));
        }

        self.input_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Attack input shader bind group"),
            layout: &self.input_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.energies_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.node_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.successor_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.changed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.work_size_buf.as_entire_binding(),
                },
            ],
        });
        self.minima_bind_group =  device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Attack minima bind group"),
            layout: &self.minima_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.minima_buf.as_entire_binding(),
                },
            ],
        });
    }

    fn collect_data(
        &mut self,
        game: &EnergyGame,
    ) -> (Vec<NodeOffsetAtk>, Vec<u32>, EnergyArray) {
        let mut energies = Array2::zeros((0, game.graph.get_conf().energy_size() as usize));
        let mut successor_offsets = Vec::new();
        let mut node_offsets = Vec::new();

        let mut visiting = FxHashSet::default();
        let max_size = self.gpu.device.limits().max_storage_buffer_binding_size as usize;
        for &node in &self.visit_list {
            node_offsets.push(NodeOffsetAtk {
                node,
                offset: energies.nrows() as u32,
                successor_offsets_idx: successor_offsets.len() as u32,
            });
            for &successor in game.graph.adj(node) {
                successor_offsets.push(energies.nrows() as u32);
                energies.append(Axis(0), game.energies[successor as usize].view()).unwrap();
            }
            // Last "successor" is always the own node in the attack case
            successor_offsets.push(energies.nrows() as u32);
            energies.append(Axis(0), game.energies[node as usize].view()).unwrap();

            // Check limits
            let node_offsets_size = mem::size_of_val(&node_offsets);
            let successor_offsets_size = mem::size_of_val(&successor_offsets);
            let energies_size = energies.len() * mem::size_of::<u32>();
            let workgroup_count = (energies.nrows() as u32).div_ceil(WORKGROUP_SIZE);
            if workgroup_count > self.gpu.device.limits().max_compute_workgroups_per_dimension ||
                energies_size > max_size ||
                node_offsets_size > max_size ||
                successor_offsets_size > max_size
            {
                // Mark last node offset as final cap
                node_offsets.last_mut().unwrap().node = u32::MAX;
                break;
            }

            visiting.insert(node);
        }

        self.visit_list = &self.visit_list - &visiting;

        if self.visit_list.is_empty() {
            // Last offset does not correspond to another starting node, mark with u32::MAX
            node_offsets.push(NodeOffsetAtk {
                node: u32::MAX,
                offset: energies.nrows() as u32,
                successor_offsets_idx: successor_offsets.len() as u32,
            });
            successor_offsets.push(energies.nrows() as u32);
        }
        let earray = EnergyArray::from_array(energies, game.graph.get_conf());
        trace!("Attack data: {:?}\n{:?}\n{}", node_offsets, successor_offsets, earray);
        (node_offsets, successor_offsets, earray)
    }

    #[inline]
    fn prepare_run(&mut self, game: &EnergyGame) {
        let data = self.collect_data(game);
        self.update(data.0, data.1, data.2);
    }

    fn compute_pass(&self, encoder: &mut wgpu::CommandEncoder, graph_bind_group: &wgpu::BindGroup) {
        let n_energies = self.node_offsets.last().map(|n| n.offset).unwrap_or_default();
        let workgroup_count = n_energies.div_ceil(WORKGROUP_SIZE);
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Attack Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.update_pipeline);
            cpass.set_bind_group(0, &self.input_bind_group, &[]);
            cpass.set_bind_group(1, graph_bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);

            cpass.set_pipeline(&self.minima_pipeline);
            cpass.set_bind_group(1, &self.minima_bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.minima_buf, 0, &self.minima_staging_buf, 0, self.minima_size());
        encoder.copy_buffer_to_buffer(
            &self.energies_buf, 0, &self.energies_staging_buf, 0, self.energies_size());
        encoder.copy_buffer_to_buffer(
            &self.changed_buf, 0, &self.changed_staging_buf, 0, self.changed_size());
    }

    const MAPPED_BUFS: usize = 3;

    fn map_buffers(&self, sender: &Sender<result::Result<(), wgpu::BufferAsyncError>>) -> usize {
        if self.node_offsets.len() <= 1 {
            return 0;
        }
        let minima_buffer_slice = self.minima_staging_buf.slice(.. self.minima_size());
        let energies_buffer_slice = self.energies_staging_buf.slice(.. self.energies_size());
        let changed_buffer_slice = self.changed_staging_buf.slice(.. self.changed_size());
        let sender0 = sender.clone();
        minima_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender0.try_send(v).expect("Channel should be writable");
        });
        let sender1 = sender.clone();
        energies_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender1.try_send(v).expect("Channel should be writable");
        });
        let sender2 = sender.clone();
        changed_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender2.try_send(v).expect("Channel should be writable");
        });
        Self::MAPPED_BUFS
    }

    fn process_results(&mut self, game: &mut EnergyGame) -> Vec<u32> {
        if self.node_offsets.len() <= 1 {
            return Vec::new();
        }
        let minima_data = self.minima_staging_buf.slice(.. self.minima_size()).get_mapped_range();
        let minima: &[MinFlags] = bytemuck::cast_slice(&minima_data);

        let changed_data = self.changed_staging_buf.slice(.. self.changed_size()).get_mapped_range();
        let changed: &[u32] = bytemuck::cast_slice(&changed_data);

        let energies_data = self.energies_staging_buf.slice(.. self.energies_size()).get_mapped_range();
        let energies_slice: &[u32] = bytemuck::cast_slice(&energies_data);
        let energy_size = game.graph.conf.energy_size() as usize;
        let n_energies = energies_slice.len() / energy_size;
        let energies_array = ArrayView2::from_shape((n_energies, energy_size), energies_slice)
            .expect("Energy array has invalid shape");

        if log_enabled!(Trace) {
            let mut msg = "Attack Minima:    ".to_string();
            for minima_chunk in minima {
                msg.push_str(&format!("{:064b} ", minima_chunk.reverse_bits()));
            }
            trace!("{}", msg);
            let energies = EnergyArray::from_array(energies_array.to_owned(), game.graph.get_conf());
            trace!("Attack Energies:\n{}", energies);
        }

        let mut changed_nodes = Vec::new();
        debug_assert!((self.node_offsets.len() - 1) <= changed.len(),
            "There should be a `changed` flag for every processed node");
        for (node_window, _) in self.node_offsets.windows(2)
            .zip(changed)
            .filter(|(_, &changed)| changed != 0)
        {
            let cur = node_window[0];
            let next_offset = node_window[1].offset as usize;
            let indices: Vec<usize> = (cur.offset as usize..next_offset)
                .filter(|i| minima[i / MinFlags::BITS as usize] & (1 << (i % MinFlags::BITS as usize)) != 0)
                .collect();

            // Write new, filtered energies
            let new_array = energies_array.select(Axis(0), indices.as_slice());
            game.energies[cur.node as usize] = EnergyArray::from_array(new_array, game.graph.get_conf());
            changed_nodes.push(cur.node);
        }

        // Unmap buffers
        drop(minima_data);
        self.minima_staging_buf.unmap();
        drop(energies_data);
        self.energies_staging_buf.unmap();
        drop(changed_data);
        self.changed_staging_buf.unmap();
        changed_nodes
    }
}


pub struct GPURunner<'a> {
    game: &'a mut EnergyGame,

    gpu: Rc<GPUCommon>,
    graph_bind_group: wgpu::BindGroup,

    atk_shader: AttackShader,
    defiter_shader: DefendIterShader,
}

impl<'a> GPURunner<'a> {

    pub async fn with_game(game: &'a mut EnergyGame) -> Result<GPURunner<'a>> {
        let conf = game.graph.get_conf();
        let gpu_common = Rc::new(GPUCommon::new().await?);
        let (graph_bind_group_layout, graph_bind_group) = game.graph.bind_group(&gpu_common.device);
        let preprocessor = make_replacements(conf);

        let atk_shader = AttackShader::new(Rc::clone(&gpu_common), conf, &graph_bind_group_layout, &preprocessor);
        let defiter_shader = DefendIterShader::new(Rc::clone(&gpu_common), conf, &graph_bind_group_layout, &preprocessor);

        Ok(GPURunner {
            game,
            gpu: gpu_common,
            graph_bind_group,

            atk_shader,
            defiter_shader,
        })
    }

    fn initialize_visit_lists(&mut self) {
        // We need to clone to_reach in order to call changed_nodes with &mut self
        let to_reach = self.game.to_reach.clone();
        self.changed_nodes(&to_reach);
    }

    fn changed_nodes(&mut self, nodes: &[u32]) {
        for &v in nodes {
            // Start with parent nodes of final points
            for &w in self.game.graph.reverse(v) {
                if self.game.graph.attacker_pos[w as usize] {
                    self.atk_shader.visit_list.insert(w);
                } else {
                    self.defiter_shader.visit_list.entry(w)
                        .or_insert(DefendIterShader::DEFAULT_SUPREMA_MEMORY);
                }
            }
        }
    }

    pub async fn execute_gpu(&mut self) -> Result<()> {
        self.initialize_visit_lists();
        loop {
            self.atk_shader.prepare_run(self.game);
            self.defiter_shader.prepare_run(self.game);
            debug!("Number of attack nodes: {}", self.atk_shader.node_offsets.len() - 1);
            debug!("Number of defend (iterative) nodes: {}", self.defiter_shader.node_offsets.len() - 1);

            let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Algorithm iteration encoder")
            });
            self.atk_shader.compute_pass(&mut encoder, &self.graph_bind_group);
            self.defiter_shader.compute_pass(&mut encoder, &self.graph_bind_group);

            // Submit command encoder for processing by GPU
            self.gpu.queue.submit(Some(encoder.finish()));

            const MAPS: usize = AttackShader::MAPPED_BUFS + DefendIterShader::MAPPED_BUFS;
            let (sender, receiver) = channel(MAPS);
            let wait_for = self.atk_shader.map_buffers(&sender) + 
                self.defiter_shader.map_buffers(&sender);

            // Wait for the GPU to finish work
            self.gpu.device.poll(wgpu::Maintain::Wait);

            for _ in 0..wait_for {
                receiver.receive().await.expect("Channel should not be closed")?;
            }

            let changed = self.atk_shader.process_results(self.game);
            self.changed_nodes(&changed);
            let changed = self.defiter_shader.process_results(self.game);
            self.changed_nodes(&changed);

            if  self.atk_shader.visit_list.is_empty() &&
                self.defiter_shader.visit_list.is_empty()
            {
                // Nothing was updated, we are done.
                break;
            }
        }
        Ok(())
    }
}
