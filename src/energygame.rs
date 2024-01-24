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
use gpuutils::{GPUCommon, bgl_entry, buffer_fits, GPUGraph};
use shadergen::{ShaderPreproc, make_replacements,
    build_attack, build_defend_update, build_defend_direct, build_defend_iterative};

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
        let weights = graph.weights.iter().map(|array| array.into())
            .collect();
        Self {
            conf: graph.conf,
            adj: graph.adj,
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

        let reverse = make_reverse(&deserialized.adj);
        let weights: std::result::Result<Vec<UpdateArray>, Self::Error> = deserialized.weights
            .iter()
            .map(|list| UpdateArray::from_conf(list.as_slice(), deserialized.conf))
            .collect();
        Ok(GameGraph {
            reverse,
            adj: deserialized.adj,
            weights: weights?,
            attacker_pos: deserialized.attacker_pos,
            conf: deserialized.conf,
        })
    }
}

pub fn make_reverse(adj: &Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    let mut reverse = vec![vec![]; adj.len()];
    for (from, adj) in adj.iter().enumerate() {
        for to in adj {
            reverse[*to as usize].push(from as u32);
        }
    }
    reverse
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "SerdeGameGraph", into = "SerdeGameGraph")]
pub struct GameGraph {
    pub adj: Vec<Vec<u32>>,
    pub reverse: Vec<Vec<u32>>,
    // One weight per edge, grouped by start node
    pub weights: Vec<UpdateArray>,
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
        let weights = raw_weights.iter().map(|upd_list| {
                UpdateArray::from_conf(upd_list.as_slice(), conf).unwrap()
            })
            .collect();

        Self {
            adj,
            reverse,
            weights,
            attacker_pos,
            conf,
        }
    }

    pub fn empty(conf: EnergyConf) -> Self {
        Self {
            adj: Vec::new(),
            reverse: Vec::new(),
            weights: Vec::new(),
            attacker_pos: Vec::new(),
            conf,
        }
    }

    pub fn n_vertices(&self) -> u32 {
        self.adj.len() as u32
    }

    pub fn get_conf(&self) -> EnergyConf {
        self.conf
    }
}

impl GPUGraph for GameGraph {
    type Weight = u8;

    fn csr(&self) -> (Vec<u32>, Vec<u32>, Vec<Self::Weight>) {
        let column_indices = self.adj.iter()
            .flatten()
            .copied()
            .collect();
        let weights = self.weights.iter()
            .flat_map(|e| e.data())
            .copied()
            .collect();
        // Cumulative sum of all list lengths. 0 is prepended manually
        let row_offsets = iter::once(0).chain(
            self.adj.iter()
            .scan(0, |state, adj| {
                *state += adj.len() as u32;
                Some(*state)
            }))
            .collect();
        (column_indices, row_offsets, weights)
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
            .filter(|&v| !graph.attacker_pos[v as usize] && graph.adj[v as usize].is_empty())
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
#[derive(Clone, Copy, Debug)]
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

    // Number of bytes needed to store `n` bitflags.
    // Rounded up to the next multiple of 8 (64 bits).
    fn minima_size(n: usize) -> u64 {
        (((n as i64 - 1) / 64 + 1) * 8) as u64
    }

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

struct DefendDirectShader {
    gpu: Rc<GPUCommon>,
    visit_list: FxHashSet<u32>,

    energies: EnergyArray,
    energies_buf: Buffer,
    node_offsets: Vec<NodeOffsetDef>,
    node_offsets_buf: Buffer,
    successor_offsets_buf: Buffer,
    sup_buf: Buffer,
    sup_staging_buf: Buffer,
    minima_buf: Buffer,
    minima_staging_buf: Buffer,

    input_bind_group: wgpu::BindGroup,
    input_bind_group_layout: wgpu::BindGroupLayout,
    update_pipeline: wgpu::ComputePipeline,

    suprema_bind_group: wgpu::BindGroup,
    suprema_bind_group_layout: wgpu::BindGroupLayout,
    combine_pipeline: wgpu::ComputePipeline,

    minima_pipeline: wgpu::ComputePipeline,
}

impl PlayerShader for DefendDirectShader {
    fn name() -> &'static str { "Defend Direct" }
}

impl DefendDirectShader {
    fn new(
        gpu: Rc<GPUCommon>,
        conf: EnergyConf,
        graph_bind_group_layout: &wgpu::BindGroupLayout,
        preprocessor: &ShaderPreproc,
    ) -> DefendDirectShader {
        let node_offsets_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Defend Direct Node offsets storage buffer initial"),
            size: INITIAL_CAPACITY * mem::size_of::<NodeOffsetDef>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let energies_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Defend Direct Energies storage buffer initial"),
            size: INITIAL_CAPACITY * u64::from(conf.energy_size()) * mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let successor_offsets_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Defend Direct Successor offsets storage buffer initial"),
            size: INITIAL_CAPACITY * mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sup_bytes = INITIAL_CAPACITY * u64::from(conf.energy_size()) * mem::size_of::<u32>() as u64;
        let (sup_buf, sup_staging_buf) = Self::new_output_buf(
            &gpu.device, sup_bytes, Some("Suprema"));

        let (minima_buf, minima_staging_buf) = Self::new_output_buf(
            &gpu.device, INITIAL_CAPACITY, Some("Minima flags"));

        let input_bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Defend Direct common input bind group layout"),
            entries: &[
                bgl_entry(0, false), // energies, writable
                bgl_entry(1, true),  // node offsets
                bgl_entry(2, true),  // successor offsets
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
            ],
        });

        // A bind group can have at most 4 storage buffers, split across 2 bind groups.
        let suprema_bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Defend Direct suprema of combinations shader bind group layout 1"),
            entries: &[
                bgl_entry(0, false), // suprema, writable
                bgl_entry(1, false), // minima, writable
            ],
        });
        let suprema_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Defend Direct combinations shader bind group (inital)"),
            layout: &suprema_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sup_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: minima_buf.as_entire_binding(),
                },
            ],
        });

        let update_pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Defend Direct suprema of combinations pipeline layout"),
            bind_group_layouts: &[&input_bind_group_layout, &graph_bind_group_layout],
            push_constant_ranges: &[],
        });
        let update_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Defend Direct energy update shader module"),
            source: build_defend_update(preprocessor),
        });
        let update_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Defend Direct energy update pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_shader,
            entry_point: "main",
        });

        let combine_pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Defend Direct suprema of combinations pipeline layout"),
            bind_group_layouts: &[&input_bind_group_layout, &suprema_bind_group_layout],
            push_constant_ranges: &[],
        });
        let combine_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Defend Direct suprema of combinations shader module"),
            source: build_defend_direct(preprocessor),
        });
        let combine_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Defend Direct suprema of combinations pipeline"),
            layout: Some(&combine_pipeline_layout),
            module: &combine_shader,
            entry_point: "main",
        });

        let minima_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Defend Direct minima pipeline"),
            layout: Some(&combine_pipeline_layout),
            module: &combine_shader,
            entry_point: "minimize",
        });

        DefendDirectShader {
            gpu,
            visit_list: FxHashSet::default(),
            energies: EnergyArray::empty(conf),
            energies_buf,
            node_offsets: Vec::new(),
            node_offsets_buf,
            successor_offsets_buf,
            sup_buf,
            sup_staging_buf,
            minima_buf,
            minima_staging_buf,

            input_bind_group,
            input_bind_group_layout,
            update_pipeline,

            suprema_bind_group,
            suprema_bind_group_layout,
            combine_pipeline,

            minima_pipeline,
        }
    }

    fn update(&mut self,
        node_offsets: Vec<NodeOffsetDef>,
        successor_offsets: Vec<u32>,
        energies: EnergyArray,
    ) {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;

        if energies.view().len() * std::mem::size_of::<u32>() > self.energies_buf.size() as usize {
            self.energies_buf = Self::get_energies_buf(device, &energies);
        } else {
            queue.write_buffer(&self.energies_buf, 0, energies.data());
        }
        self.energies = energies;

        // Make sure the node offsets buffer always has exactly the right size
        if node_offsets.len() * std::mem::size_of::<NodeOffsetDef>() != self.node_offsets_buf.size() as usize {
            self.node_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Defend Direct node offsets storage buffer"),
                contents: bytemuck::cast_slice(&node_offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&self.node_offsets_buf, 0, bytemuck::cast_slice(&node_offsets));
        }
        self.node_offsets = node_offsets;

        if !buffer_fits(&successor_offsets, &self.successor_offsets_buf) {
            self.successor_offsets_buf = Self::get_successor_offsets_buf(device, &successor_offsets);
        } else {
            queue.write_buffer(&self.successor_offsets_buf, 0, bytemuck::cast_slice(&successor_offsets));
        }

        let sup_size: u64 = self.node_offsets.last().expect("Even if visit list is empty, node offsets has one entry")
            .sup_offset.into();
        let sup_bytes = sup_size * u64::from(self.energies.get_conf().energy_size()) * std::mem::size_of::<u32>() as u64;
        if sup_bytes > self.sup_buf.size() {
            (self.sup_buf, self.sup_staging_buf) = Self::new_output_buf(
                device, sup_bytes, Some("Suprema"));
        }

        let minima_capacity = Self::minima_size(sup_size as usize);
        if minima_capacity > self.minima_buf.size() {
            (self.minima_buf, self.minima_staging_buf) = Self::new_output_buf(
                device, minima_capacity, Some("Minima flags"));
        }

        self.input_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Defend Direct common input bind group (updated)"),
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
            ],
        });

        self.suprema_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Defend Direct suprema of combinations bind group (updated)"),
            layout: &self.suprema_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sup_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.minima_buf.as_entire_binding(),
                },
            ],
        });
    }

    fn collect_data(
        &mut self,
        game: &EnergyGame
    ) -> (Vec<NodeOffsetDef>, Vec<u32>, EnergyArray) {
        let mut energies = Array2::zeros((0, game.graph.get_conf().energy_size() as usize));
        let mut successor_offsets = Vec::new();
        let mut node_offsets = Vec::new();
        let mut sup_count = 0;

        let max_size = self.gpu.device.limits().max_storage_buffer_binding_size as usize - 256;
        let max_wg = self.gpu.device.limits().max_compute_workgroups_per_dimension;
        let mut visiting = FxHashSet::default();
        for &node in &self.visit_list {
            visiting.insert(node);
            let snode = node as usize;

            // If any successor has no energies associated yet, skip this node
            if game.graph.adj[snode].iter()
                .any(|&suc| game.energies[suc as usize].is_empty())
            { continue }

            node_offsets.push(NodeOffsetDef {
                node,
                successor_offsets_idx: successor_offsets.len() as u32,
                energy_offset: energies.nrows() as u32,
                sup_offset: sup_count,
            });
            let mut cur_sup_count: u32 = 1;
            for &successor in &game.graph.adj[snode] {
                successor_offsets.push(energies.nrows() as u32);
                let successor_energies = &game.energies[successor as usize];
                energies.append(Axis(0), successor_energies.view()).unwrap();
                cur_sup_count = cur_sup_count.saturating_mul(successor_energies.n_energies() as u32);
            }
            sup_count = sup_count.saturating_add(cur_sup_count);

            // Check limits
            let node_offsets_size = mem::size_of_val(&node_offsets);
            let successor_offsets_size = mem::size_of_val(&successor_offsets);
            let energies_size = energies.len() * mem::size_of::<u32>();
            let upd_wg_count = (energies.nrows() as u32).div_ceil(WORKGROUP_SIZE);
            let sup_wg_count = sup_count.div_ceil(WORKGROUP_SIZE);
            let sup_size = (sup_count * game.graph.get_conf().energy_size()) as usize * mem::size_of::<u32>();
            if upd_wg_count > max_wg || // Update workgroups
                sup_wg_count > max_wg || // Suprema workgroups
                sup_size > max_size ||
                energies_size > max_size ||
                node_offsets_size > max_size ||
                successor_offsets_size > max_size
            {
                // This node doesn't fit, mark last node offset as final cap
                node_offsets.last_mut().unwrap().node = u32::MAX;
                visiting.remove(&node);
                break;
            }
        }

        self.visit_list = &self.visit_list - &visiting;

        if self.visit_list.is_empty() {
            // Last offset does not correspond to another starting node, mark with u32::MAX
            node_offsets.push(NodeOffsetDef {
                node: u32::MAX,
                successor_offsets_idx: successor_offsets.len() as u32,
                energy_offset: energies.nrows() as u32,
                sup_offset: sup_count,
            });
            successor_offsets.push(energies.nrows() as u32);
        }

        let earray = EnergyArray::from_array(energies, game.graph.get_conf());
        trace!("Defend Direct data: {:?}\n{:?}\n{}", node_offsets, successor_offsets, earray);

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
                label: Some("Defend Direct energy update compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.update_pipeline);
            cpass.set_bind_group(0, &self.input_bind_group, &[]);
            cpass.set_bind_group(1, graph_bind_group, &[]);

            let n_energies = self.node_offsets.last().map(|n| n.energy_offset).unwrap_or_default();
            let update_workgroup_count = n_energies.div_ceil(WORKGROUP_SIZE);
            cpass.dispatch_workgroups(update_workgroup_count, 1, 1);
        }
        let n_sup = self.node_offsets.last().expect("Even if visit list is empty, node offsets has one entry")
            .sup_offset;
        let workgroup_count = n_sup.div_ceil(WORKGROUP_SIZE);
        { // Compute pass for taking suprema of combinations
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Defend Direct suprema of combinations compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.combine_pipeline);
            cpass.set_bind_group(0, &self.input_bind_group, &[]);
            cpass.set_bind_group(1, &self.suprema_bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);

            cpass.set_pipeline(&self.minima_pipeline);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.minima_buf, 0, &self.minima_staging_buf, 0, self.minima_buf.size());
        encoder.copy_buffer_to_buffer(
            &self.sup_buf, 0, &self.sup_staging_buf, 0, self.sup_buf.size());
    }

    fn map_buffers(&self, sender: &Sender<result::Result<(), wgpu::BufferAsyncError>>) {
        let minima_buffer_slice = self.minima_staging_buf.slice(..);
        let sup_buffer_slice = self.sup_staging_buf.slice(..);
        let sender0 = sender.clone();
        minima_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender0.try_send(v).expect("Channel should be writable");
        });
        let sender1 = sender.clone();
        sup_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender1.try_send(v).expect("Channel should be writable");
        });
    }

    fn process_results(&mut self, game: &mut EnergyGame) -> Vec<u32> {
        let minima_data = self.minima_staging_buf.slice(..).get_mapped_range();
        let minima: &[u64] = bytemuck::cast_slice(&minima_data);

        let sup_data = self.sup_staging_buf.slice(..).get_mapped_range();
        let sup_vec: Vec<u32> = bytemuck::cast_slice(&sup_data).to_vec();
        let energy_size = game.graph.get_conf().energy_size() as usize;
        let n_sup = sup_vec.len() / energy_size;
        let sup_array = Array2::from_shape_vec((n_sup, energy_size), sup_vec).expect("Suprema array has invalid shape");
        let suprema = EnergyArray::from_array(sup_array, game.graph.get_conf());

        if log_enabled!(Trace) {
            let mut msg = "Defend Direct Minima:   ".to_string();
            for minima_chunk in minima {
                msg.push_str(&format!("{:064b} ", minima_chunk.reverse_bits()));
            }
            trace!("{}", msg);
            trace!("Defend Direct Suprema:\n{}", suprema);
        }

        const MINIMA_SIZE: usize = u64::BITS as usize;

        let mut changed_nodes = Vec::new();
        for node_window in self.node_offsets.windows(2) {
            let cur = node_window[0];
            let next_offset = node_window[1].sup_offset;
            let prev = &game.energies[cur.node as usize];

            let indices: Vec<usize> = (cur.sup_offset as usize..next_offset as usize)
                .filter(|i| minima[i / MINIMA_SIZE] & (1 << (i % MINIMA_SIZE)) != 0)
                .collect();

            let new_array = suprema.view().select(Axis(0), indices.as_slice());
            let new_energies = EnergyArray::from_array(new_array, game.graph.get_conf());

            if &new_energies != prev {
                // Write new, filtered energies
                game.energies[cur.node as usize] = new_energies;
                changed_nodes.push(cur.node);
            }
        }

        // Unmap buffers
        drop(minima_data);
        self.minima_staging_buf.unmap();
        drop(sup_data);
        self.sup_staging_buf.unmap();
        changed_nodes
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

        let input_bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Defend Iterative common input bind group layout"),
            entries: &[
                bgl_entry(0, false), // energies, writable
                bgl_entry(1, true),  // node offsets
                bgl_entry(2, true),  // successor offsets
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

            input_bind_group,
            input_bind_group_layout,
            update_pipeline,

            suprema_bind_group,
            suprema_bind_group_layout,
            combine_pipeline,
        }
    }

    fn update(&mut self,
        node_offsets: Vec<NodeOffsetDef>,
        successor_offsets: Vec<u32>,
        energies: EnergyArray,
    ) {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;

        if energies.view().len() * mem::size_of::<u32>() > self.energies_buf.size() as usize {
            self.energies_buf = Self::get_energies_buf(device, &energies);
        } else {
            queue.write_buffer(&self.energies_buf, 0, energies.data());
        }
        self.energies = energies;

        // Make sure the node offsets buffer always has exactly the right size
        if node_offsets.len() * mem::size_of::<NodeOffsetDef>() != self.node_offsets_buf.size() as usize {
            self.node_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Defend Iterative node offsets storage buffer"),
                contents: bytemuck::cast_slice(&node_offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&self.node_offsets_buf, 0, bytemuck::cast_slice(&node_offsets));
        }
        self.node_offsets = node_offsets;

        if !buffer_fits(&successor_offsets, &self.successor_offsets_buf) {
            self.successor_offsets_buf = Self::get_successor_offsets_buf(device, &successor_offsets);
        } else {
            queue.write_buffer(&self.successor_offsets_buf, 0, bytemuck::cast_slice(&successor_offsets));
        }

        let sup_size: u64 = self.node_offsets.last().expect("Even if visit list is empty, node offsets has one entry")
            .sup_offset.into();
        let sup_bytes = sup_size * u64::from(self.energies.get_conf().energy_size()) * mem::size_of::<u32>() as u64;
        if sup_bytes > self.sup_buf.size() {
            (self.sup_buf, self.sup_staging_buf) = Self::new_output_buf(
                device, sup_bytes, Some("Suprema"));
        }

        let status_bytes = ((self.node_offsets.len() - 1) * mem::size_of::<i32>()) as u64;
        if status_bytes > self.status_buf.size() {
            (self.status_buf, self.status_staging_buf) = Self::new_output_buf(
                device, status_bytes, Some("Intersection status"));
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
            let snode = node as usize;

            // If any successor has no energies associated yet, skip this node
            if game.graph.adj[snode].iter()
                .any(|&suc| game.energies[suc as usize].is_empty())
            { continue }

            node_offsets.push(NodeOffsetDef {
                node,
                successor_offsets_idx: successor_offsets.len() as u32,
                energy_offset: energies.nrows() as u32,
                sup_offset: sup_count,
            });
            for &successor in &game.graph.adj[snode] {
                successor_offsets.push(energies.nrows() as u32);
                energies.append(Axis(0), game.energies[successor as usize].view()).unwrap();
            }
            // Allocate the requested amount of suprema memory for this node
            sup_count = sup_count.saturating_add(mem as u32);

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
                label: Some("Defend Iterative energy update compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.update_pipeline);
            cpass.set_bind_group(0, &self.input_bind_group, &[]);
            cpass.set_bind_group(1, graph_bind_group, &[]);

            let n_energies = self.node_offsets.last().map(|n| n.energy_offset).unwrap_or_default();
            let update_workgroup_count = n_energies.div_ceil(WORKGROUP_SIZE);
            cpass.dispatch_workgroups(update_workgroup_count, 1, 1);
        }
        { // Compute pass for taking suprema of combinations
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Defend Iterative suprema of combinations compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.combine_pipeline);
            cpass.set_bind_group(0, &self.input_bind_group, &[]);
            cpass.set_bind_group(1, &self.suprema_bind_group, &[]);
            // One workgroup for each node
            cpass.dispatch_workgroups(self.node_offsets.len().max(1) as u32 - 1, 1, 1);
        }
        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.status_buf, 0, &self.status_staging_buf, 0, self.status_buf.size());
        encoder.copy_buffer_to_buffer(
            &self.sup_buf, 0, &self.sup_staging_buf, 0, self.sup_buf.size());
    }

    fn map_buffers(&self, sender: &Sender<result::Result<(), wgpu::BufferAsyncError>>) {
        let status_buffer_slice = self.status_staging_buf.slice(..);
        let sup_buffer_slice = self.sup_staging_buf.slice(..);
        let sender0 = sender.clone();
        status_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender0.try_send(v).expect("Channel should be writable");
        });
        let sender1 = sender.clone();
        sup_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender1.try_send(v).expect("Channel should be writable");
        });
    }

    fn process_results(&mut self, game: &mut EnergyGame) -> Vec<u32> {
        let status_data = self.status_staging_buf.slice(..).get_mapped_range();
        let status: &[i32] = bytemuck::cast_slice(&status_data);

        let sup_data = self.sup_staging_buf.slice(..).get_mapped_range();
        let sup_vec: Vec<u32> = bytemuck::cast_slice(&sup_data).to_vec();
        let energy_size = game.graph.get_conf().energy_size() as usize;
        let n_sup = sup_vec.len() / energy_size;
        let sup_array = ArrayView2::from_shape((n_sup, energy_size), &sup_vec)
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

    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
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

        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Attack energy bind group layout"),
            entries: &[
                bgl_entry(0, false), // energies, writable
                bgl_entry(1, true),  // node offsets
                bgl_entry(2, true),  // successor offsets
                bgl_entry(3, false), // minima, writable
            ],
        });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} bind group", Self::name())),
            layout: &bind_group_layout,
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
                    resource: minima_buf.as_entire_binding(),
                },
            ],
        });

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Attack shader module"),
            source: build_attack(preprocessor),
        });
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Attack pipeline layout"),
            bind_group_layouts: &[&bind_group_layout, &graph_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Attack compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        let minima_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Attack minima pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
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

            bind_group,
            bind_group_layout,
            pipeline,
            minima_pipeline,
        }
    }

    fn update(&mut self,
        node_offsets: Vec<NodeOffsetAtk>,
        successor_offsets: Vec<u32>,
        energies: EnergyArray,
    ) {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;

        if energies.view().len() * mem::size_of::<u32>() > self.energies_buf.size() as usize {
            self.energies_buf = Self::get_energies_buf(device, &energies);
            self.energies_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Energies staging buffer", Self::name())),
                size: self.energies_buf.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        } else {
            queue.write_buffer(&self.energies_buf, 0, energies.data());
        }
        self.energies = energies;

        // Make sure the node offsets buffer always has exactly the right size
        if node_offsets.len() * mem::size_of::<NodeOffsetAtk>() != self.node_offsets_buf.size() as usize {
            self.node_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Node offsets storage buffer", Self::name())),
                contents: bytemuck::cast_slice(&node_offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&self.node_offsets_buf, 0, bytemuck::cast_slice(&node_offsets));
        }
        self.node_offsets = node_offsets;

        if !buffer_fits(&successor_offsets, &self.successor_offsets_buf) {
            self.successor_offsets_buf = Self::get_successor_offsets_buf(device, &successor_offsets);
        } else {
            queue.write_buffer(&self.successor_offsets_buf, 0, bytemuck::cast_slice(&successor_offsets));
        }

        let minima_capacity = Self::minima_size(self.energies.n_energies());
        if minima_capacity > self.minima_buf.size() {
            (self.minima_buf, self.minima_staging_buf) = Self::new_output_buf(
                device, minima_capacity, Some("Minima flags"));
        }

        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Main shader bind group", Self::name())),
            layout: &self.bind_group_layout,
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
            for &successor in &game.graph.adj[node as usize] {
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
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_bind_group(1, graph_bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);

            cpass.set_pipeline(&self.minima_pipeline);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.minima_buf, 0, &self.minima_staging_buf, 0, self.minima_buf.size());
        encoder.copy_buffer_to_buffer(
            &self.energies_buf, 0, &self.energies_staging_buf, 0, self.energies_buf.size());
    }

    fn map_buffers(&self, sender: &Sender<result::Result<(), wgpu::BufferAsyncError>>) {
        let minima_buffer_slice = self.minima_staging_buf.slice(..);
        let energies_buffer_slice = self.energies_staging_buf.slice(..);
        let sender0 = sender.clone();
        minima_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender0.try_send(v).expect("Channel should be writable");
        });
        let sender1 = sender.clone();
        energies_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender1.try_send(v).expect("Channel should be writable");
        });
    }

    fn process_results(&mut self, game: &mut EnergyGame) -> Vec<u32> {
        let minima_data = self.minima_staging_buf.slice(..).get_mapped_range();
        let minima: &[u64] = bytemuck::cast_slice(&minima_data);

        let energies_data = self.energies_staging_buf.slice(..).get_mapped_range();
        let energies_vec: Vec<u32> = bytemuck::cast_slice(&energies_data).to_vec();
        let energy_size = game.graph.conf.energy_size() as usize;
        let n_energies = energies_vec.len() / energy_size;
        let energies_array = Array2::from_shape_vec((n_energies, energy_size), energies_vec)
            .expect("Energy array has invalid shape");
        let energies = EnergyArray::from_array(energies_array, game.graph.get_conf());

        if log_enabled!(Trace) {
            let mut msg = "Attack Minima:    ".to_string();
            for minima_chunk in minima {
                msg.push_str(&format!("{:064b} ", minima_chunk.reverse_bits()));
            }
            trace!("{}", msg);
            trace!("Attack Energies:\n{}", energies);
        }

        const MINIMA_SIZE: u32 = u64::BITS;

        let mut changed_nodes = Vec::new();
        for node_window in self.node_offsets.windows(2) {
            let cur = node_window[0];
            let next_offset = node_window[1].offset;
            let width = next_offset - cur.offset;
            let n_prev = game.energies[cur.node as usize].n_energies() as u32;
            let n_new = width - n_prev;
            debug_assert!(n_prev <= width);

            let mut changed = false;
            // If energies didn't change, all newly compared energies will be filtered out
            if n_new > 0 {
                for minima_idx in (cur.offset / MINIMA_SIZE) ..= ((cur.offset + n_new - 1) / MINIMA_SIZE) {
                    // What position in the minima u64 the node starts at
                    let shift_start = cur.offset.max(minima_idx * MINIMA_SIZE) % MINIMA_SIZE;
                    // ... and ends at
                    let pad_right = (cur.offset + n_new - 1).min((minima_idx + 1) * MINIMA_SIZE - 1) % MINIMA_SIZE;
                    let shift_end = MINIMA_SIZE - 1 - pad_right;
                    let mask = (u64::MAX << (shift_start + shift_end)) >> shift_end;

                    let minima_chunk = minima[minima_idx as usize];
                    if minima_chunk & mask != 0 {
                        changed = true;
                        break;
                    }
                }
            }

            if changed {
                let indices: Vec<usize> = (cur.offset as usize..next_offset as usize)
                    .filter(|i| minima[i / MINIMA_SIZE as usize] & (1 << (i % MINIMA_SIZE as usize)) != 0)
                    .collect();

                // Write new, filtered energies
                let new_array = energies.view().select(Axis(0), indices.as_slice());
                game.energies[cur.node as usize] = EnergyArray::from_array(new_array, game.graph.get_conf());
                changed_nodes.push(cur.node);
            }
        }

        // Unmap buffers
        drop(minima_data);
        self.minima_staging_buf.unmap();
        drop(energies_data);
        self.energies_staging_buf.unmap();
        changed_nodes
    }
}


pub struct GPURunner<'a> {
    game: &'a mut EnergyGame,

    gpu: Rc<GPUCommon>,
    graph_bind_group: wgpu::BindGroup,

    atk_shader: AttackShader,
    defdir_shader: DefendDirectShader,
    defiter_shader: DefendIterShader,
}

impl<'a> GPURunner<'a> {

    pub async fn with_game(game: &'a mut EnergyGame) -> Result<GPURunner<'a>> {
        let conf = game.graph.get_conf();
        let gpu_common = Rc::new(GPUCommon::new().await?);
        let (graph_bind_group_layout, graph_bind_group) = game.graph.bind_group(&gpu_common.device);
        let preprocessor = make_replacements(conf);

        let atk_shader = AttackShader::new(Rc::clone(&gpu_common), conf, &graph_bind_group_layout, &preprocessor);
        let defdir_shader = DefendDirectShader::new(Rc::clone(&gpu_common), conf, &graph_bind_group_layout, &preprocessor);
        let defiter_shader = DefendIterShader::new(Rc::clone(&gpu_common), conf, &graph_bind_group_layout, &preprocessor);

        Ok(GPURunner {
            game,
            gpu: gpu_common,
            graph_bind_group,

            atk_shader,
            defdir_shader,
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
            for &w in &self.game.graph.reverse[v as usize] {
                if self.game.graph.attacker_pos[w as usize] {
                    self.atk_shader.visit_list.insert(w);
                } else {
                    let combinations: u32 = self.game.graph.adj[w as usize].iter()
                        .map(|&suc| self.game.energies[suc as usize].n_energies() as u32)
                        .reduce(|acc, e| acc.saturating_mul(e))
                        .unwrap_or_default();
                    if combinations <= 64 {
                        self.defdir_shader.visit_list.insert(w);
                    } else {
                        self.defiter_shader.visit_list.entry(w)
                            .or_insert(DefendIterShader::DEFAULT_SUPREMA_MEMORY);
                    }
                }
            }
        }
    }

    pub async fn execute_gpu(&mut self) -> Result<()> {
        self.initialize_visit_lists();
        loop {
            self.atk_shader.prepare_run(self.game);
            self.defdir_shader.prepare_run(self.game);
            self.defiter_shader.prepare_run(self.game);
            debug!("Number of attack nodes: {}", self.atk_shader.node_offsets.len() - 1);
            debug!("Number of defend (direct) nodes: {}", self.defdir_shader.node_offsets.len() - 1);
            debug!("Number of defend (iterative) nodes: {}", self.defiter_shader.node_offsets.len() - 1);

            let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Algorithm iteration encoder")
            });
            self.atk_shader.compute_pass(&mut encoder, &self.graph_bind_group);
            self.defdir_shader.compute_pass(&mut encoder, &self.graph_bind_group);
            self.defiter_shader.compute_pass(&mut encoder, &self.graph_bind_group);

            // Submit command encoder for processing by GPU
            self.gpu.queue.submit(Some(encoder.finish()));

            const MAPS: usize = 6;
            let (sender, receiver) = channel(MAPS);
            self.atk_shader.map_buffers(&sender);
            self.defdir_shader.map_buffers(&sender);
            self.defiter_shader.map_buffers(&sender);

            // Wait for the GPU to finish work
            self.gpu.device.poll(wgpu::Maintain::Wait);

            for _ in 0..MAPS {
                receiver.receive().await.expect("Channel should not be closed")?;
            }

            let changed = self.atk_shader.process_results(self.game);
            self.changed_nodes(&changed);
            let changed = self.defdir_shader.process_results(self.game);
            self.changed_nodes(&changed);
            let changed = self.defiter_shader.process_results(self.game);
            self.changed_nodes(&changed);

            if  self.atk_shader.visit_list.is_empty() &&
                self.defdir_shader.visit_list.is_empty() &&
                self.defiter_shader.visit_list.is_empty()
            {
                // Nothing was updated, we are done.
                break;
            }
        }
        Ok(())
    }
}
