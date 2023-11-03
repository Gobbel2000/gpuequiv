mod types;

use std::borrow::Cow;
use std::collections::HashSet;
use std::{fmt, error, result};
use std::iter;
use std::rc::Rc;

use futures_intrusive::channel::shared::{Sender, channel};
use wgpu::{Buffer, Device, Queue};
use wgpu::util::DeviceExt;

pub use crate::types::{Energy, Update, Upd};

// Spawn 64 threads with each workgroup invocation
const WORKGROUP_SIZE: u32 = 64;
// Initial size of buffers in u32's, if there is no data yet.
// Buffers with size 0 are not allowed.
const INITIAL_CAPACITY: usize = 64;

#[derive(Debug, Clone)]
pub struct GameGraph {
    pub n_vertices: u32,
    pub adj: Vec<Vec<u32>>,
    pub reverse: Vec<Vec<u32>>,
    pub weights: Vec<Vec<Update>>,
    pub attacker_pos: Vec<bool>,
}

impl GameGraph {
    pub fn new(n_vertices: u32, edges: &[(u32, u32, Update)], attacker_pos: &[bool]) -> Self {
        let mut adj = vec![vec![]; n_vertices as usize];
        let mut reverse = vec![vec![]; n_vertices as usize];
        let mut weights = vec![vec![]; n_vertices as usize];
        for (from, to, e) in edges {
            adj[*from as usize].push(*to);
            reverse[*to as usize].push(*from);
            weights[*from as usize].push(*e);
        }

        Self {
            n_vertices,
            adj,
            reverse,
            weights,
            attacker_pos: attacker_pos.to_vec(),
        }
    }

    fn csr(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let column_indices = self.adj.iter()
            .flatten()
            .copied()
            .collect();
        let weights = self.weights.iter()
            .flatten()
            .copied()
            .map(|e| e.0)
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
    pub energies: Vec<Vec<Energy>>,
    pub to_reach: Vec<u32>,
}

impl EnergyGame {

    pub fn from_graph(graph: GameGraph) -> Self {
        let energies = vec![vec![]; graph.n_vertices as usize];
        EnergyGame { graph, energies, to_reach: vec![] }
    }

    pub fn with_reach(mut self, to_reach: Vec<u32>) -> Self {
        for v in &to_reach {
            self.energies[*v as usize].push(Energy::zero());
        }
        self.to_reach = to_reach;
        self
    }

    pub async fn get_gpu_runner(&mut self) -> Result<GPURunner> {
        GPURunner::with_game(self).await
    }

    pub async fn run(&mut self) -> Result<&[Vec<Energy>]> {
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

    // Number of u32's needed to store `n` bitflags.
    // Rounded up to the next multiple of 2 (64 bits).
    fn minima_size(n: usize) -> usize {
        (((n as isize - 1) / 64 + 1) * 2) as usize
    }

    // Construct buffer for energies
    fn get_energies_buf(device: &Device, energies: &[Energy]) -> Buffer {
        if energies.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Energies storage buffer empty", Self::name())),
                size: (INITIAL_CAPACITY * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Energies storage buffer", Self::name())),
                contents: bytemuck::cast_slice(&energies),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            })
        }
    }

    fn get_successor_offsets_buf(device: &Device, successor_offsets: &[u32]) -> Buffer {
        if successor_offsets.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Successor offsets storage buffer empty", Self::name())),
                size: (INITIAL_CAPACITY * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Successor offsets storage buffer", Self::name())),
                contents: bytemuck::cast_slice(&successor_offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        }
    }

    // Create minima buffer, with staging buffer for reading.
    // `size` is the number of u32's to allocate, meaning size*32 flags can be stored.
    fn get_minima_buf(device: &Device, size: usize) -> (Buffer, Buffer) {
        // Output flags are bit-packed with 32 bools per u32. Round up to next multiple of 64.
        // minima_capacity is measured in number of u32's.
        let minima_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Minimal energy flags storage buffer", Self::name())),
            size: (size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // For reading minima_buf on the CPU
        let minima_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Minima output staging buffer", Self::name())),
            size: minima_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (minima_buf, minima_staging_buf)
    }
}


struct DefendShader {
    gpu: Rc<GPUCommon>,
    visit_list: Vec<u32>,

    energies: Vec<Energy>,
    energies_buf: Buffer,
    node_offsets: Vec<NodeOffsetDef>,
    node_offsets_buf: Buffer,
    successor_offsets_buf: Buffer,
    sup_buf: Buffer,
    sup_staging_buf: Buffer,
    minima_buf: Buffer,
    minima_staging_buf: Buffer,

    update_bind_group: wgpu::BindGroup,
    update_bind_group_layout: wgpu::BindGroupLayout,
    update_pipeline: wgpu::ComputePipeline,

    combine_bind_groups: [wgpu::BindGroup; 2],
    combine_bind_group_layouts: [wgpu::BindGroupLayout; 2],
    combine_pipeline: wgpu::ComputePipeline,

    minima_pipeline: wgpu::ComputePipeline,
}

impl PlayerShader for DefendShader {
    fn name() -> &'static str { "Defend" }
}

impl DefendShader {
    fn new(
        gpu: Rc<GPUCommon>,
        game: &EnergyGame,
        graph_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<DefendShader> {
        let mut visit_list: Vec<u32> = Vec::new();
        for &v in &game.to_reach {
            // Start with parent nodes of final points
            for &w in &game.graph.reverse[v as usize] {
                if !game.graph.attacker_pos[w as usize] {
                    visit_list.push(w);
                }
            }
        }

        let (node_offsets, successor_offsets, energies) = Self::collect_data(&visit_list, game)?;
        let node_offsets_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Node offsets storage buffer", Self::name())),
            contents: bytemuck::cast_slice(&node_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let energies_buf = Self::get_energies_buf(&gpu.device, &energies);
        let successor_offsets_buf = Self::get_successor_offsets_buf(&gpu.device, &successor_offsets);

        let sup_size = (node_offsets.last().expect("Even if visit list is empty, node offsets has one entry")
            .sup_offset as usize)
            .max(INITIAL_CAPACITY);
        let sup_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Defend suprema of combinations storage buffer"),
            size: (sup_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sup_staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Defend suprema of combinations staging buffer"),
            size: sup_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (minima_buf, minima_staging_buf) = Self::get_minima_buf(&gpu.device,
            Self::minima_size(sup_size));

        let update_bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Defend suprema of combinations shader bind group layout"),
            entries: &[
                bgl_entry(0, false), // energies, writable
                bgl_entry(1, true),  // node offsets
                bgl_entry(2, true),  // successor offsets
            ],
        });
        let update_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} energy update bind group (initial)", Self::name())),
            layout: &update_bind_group_layout,
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
        let combine_bind_group_layouts = [
            gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Defend suprema of combinations shader bind group layout 0"),
                entries: &[
                    bgl_entry(0, true),  // energies
                    bgl_entry(1, true),  // node offsets
                    bgl_entry(2, true),  // successor offsets
                    bgl_entry(3, false), // minima, writable
                ],
            }),
            gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Defend suprema of combinations shader bind group layout 1"),
                entries: &[
                    bgl_entry(0, false), // suprema, writable
                ],
            }),
        ];
        let combine_bind_groups = [
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Defend combinations shader bind group (inital)"),
                layout: &combine_bind_group_layouts[0],
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
            }),
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Defend combinations shader bind group (inital)"),
                layout: &combine_bind_group_layouts[1],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sup_buf.as_entire_binding(),
                    },
                ],
            }),
        ];

        let update_pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Defend suprema of combinations pipeline layout"),
            bind_group_layouts: &[&update_bind_group_layout, &graph_bind_group_layout],
            push_constant_ranges: &[],
        });
        let update_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Defend energy update shader module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("defend_update.wgsl"))),
        });
        let update_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Defend energy update pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_shader,
            entry_point: "main",
        });

        let combine_pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Defend suprema of combinations pipeline layout"),
            bind_group_layouts: &[&combine_bind_group_layouts[0], &combine_bind_group_layouts[1]],
            push_constant_ranges: &[],
        });
        let combine_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Defend suprema of combinations shader module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("defend_combine.wgsl"))),
        });
        let combine_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Defend suprema of combinations pipeline"),
            layout: Some(&combine_pipeline_layout),
            module: &combine_shader,
            entry_point: "main",
        });

        let minima_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Defend minima pipeline"),
            layout: Some(&combine_pipeline_layout),
            module: &combine_shader,
            entry_point: "minimize",
        });

        Ok(DefendShader {
            gpu,
            visit_list,
            energies,
            energies_buf,
            node_offsets,
            node_offsets_buf,
            successor_offsets_buf,
            sup_buf,
            sup_staging_buf,
            minima_buf,
            minima_staging_buf,

            update_bind_group,
            update_bind_group_layout,
            update_pipeline,

            combine_bind_groups,
            combine_bind_group_layouts,
            combine_pipeline,

            minima_pipeline,
        })
    }

    fn update(&mut self,
        game: &EnergyGame,
    ) -> Result<()> {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        let (node_offsets, successor_offsets, energies) = Self::collect_data(&self.visit_list, game)?;

        if !buffer_fits(&energies, &self.energies_buf) {
            self.energies_buf = Self::get_energies_buf(&device, &energies);
        } else {
            queue.write_buffer(&self.energies_buf, 0, bytemuck::cast_slice(&energies));
        }
        self.energies = energies;

        // Make sure the node offsets buffer always has exactly the right size
        if node_offsets.len() * std::mem::size_of::<NodeOffsetAtk>() != self.node_offsets_buf.size() as usize {
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
            self.successor_offsets_buf = Self::get_successor_offsets_buf(&device, &successor_offsets);
        } else {
            queue.write_buffer(&self.successor_offsets_buf, 0, bytemuck::cast_slice(&successor_offsets));
        }

        let sup_size = self.node_offsets.last().expect("Even if visit list is empty, node offsets has one entry")
            .sup_offset as usize;
        if sup_size * std::mem::size_of::<u32>() > self.sup_buf.size() as usize {
            self.sup_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Defend suprema of combinations storage buffer"),
                size: (sup_size * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.sup_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Defend suprema of combinations staging buffer"),
                size: self.sup_buf.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        let minima_capacity = Self::minima_size(sup_size);
        if minima_capacity * std::mem::size_of::<u32>() > self.minima_buf.size() as usize {
            (self.minima_buf, self.minima_staging_buf) = Self::get_minima_buf(&device, minima_capacity);
        }

        self.update_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Defend energy update bind group (updated)"),
            layout: &self.update_bind_group_layout,
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

        self.combine_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Defend suprema of combinations bind group (updated)"),
                layout: &self.combine_bind_group_layouts[0],
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
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Defend combinations shader bind group (inital)"),
                layout: &self.combine_bind_group_layouts[1],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.sup_buf.as_entire_binding(),
                    },
                ],
            }),
        ];
        Ok(())
    }

    fn collect_data(
        visit_list: &[u32],
        game: &EnergyGame,
    ) -> Result<(Vec<NodeOffsetDef>, Vec<u32>, Vec<Energy>)> {
        let mut energies = Vec::new();
        let mut successor_offsets = Vec::new();
        let mut successor_offsets_count = 0;
        let mut node_offsets = Vec::new();
        let mut energies_count = 0;
        let mut sup_count = 0;

        for node in visit_list {
            let snode = *node as usize;

            // If any successor has no energies associated yet, skip this node
            if game.graph.adj[snode].iter()
                .any(|&suc| game.energies[suc as usize].is_empty())
            { continue }

            node_offsets.push(NodeOffsetDef {
                node: *node,
                successor_offsets_idx: successor_offsets_count,
                energy_offset: energies_count,
                sup_offset: sup_count,
            });
            let mut cur_sup_count: u32 = 1;
            for &successor in &game.graph.adj[snode] {
                let successor_energies = &game.energies[successor as usize];
                energies.extend(successor_energies);
                successor_offsets.push(energies_count);
                energies_count = energies_count.checked_add(successor_energies.len() as u32)
                    .ok_or(Error::Overflow)?;
                cur_sup_count = cur_sup_count.checked_mul(successor_energies.len() as u32)
                    .ok_or(Error::Overflow)?;
            }
            successor_offsets_count = successor_offsets_count.checked_add(game.graph.adj[snode].len() as u32)
                .ok_or(Error::Overflow)?;
            sup_count = sup_count.checked_add(cur_sup_count).ok_or(Error::Overflow)?;
        }
        successor_offsets.push(energies_count);
        // Last offset does not correspond to another starting node, mark with u32::MAX
        node_offsets.push(NodeOffsetDef {
            node: u32::MAX,
            successor_offsets_idx: successor_offsets_count,
            energy_offset: energies_count,
            sup_offset: sup_count,
        });
        println!("Defend data: {:?}\n{:?}\n{:?}", node_offsets, successor_offsets, energies);

        Ok((node_offsets, successor_offsets, energies))
    }

    fn compute_pass(&self, encoder: &mut wgpu::CommandEncoder, graph_bind_group: &wgpu::BindGroup) {
        { // Compute pass for updating energies
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Defend energy update compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.update_pipeline);
            cpass.set_bind_group(0, &self.update_bind_group, &[]);
            cpass.set_bind_group(1, graph_bind_group, &[]);

            let n_energies = self.energies.len() as u32;
            let update_workgroup_count = n_energies.div_ceil(WORKGROUP_SIZE);
            cpass.dispatch_workgroups(update_workgroup_count, 1, 1);
        }
        let n_sup = self.node_offsets.last().expect("Even if visit list is empty, node offsets has one entry")
            .sup_offset;
        let workgroup_count = n_sup.div_ceil(WORKGROUP_SIZE); 
        { // Compute pass for taking suprema of combinations
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Defend suprema of combinations compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.combine_pipeline);
            cpass.set_bind_group(0, &self.combine_bind_groups[0], &[]);
            cpass.set_bind_group(1, &self.combine_bind_groups[1], &[]);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        { // Compute pass for minimizing final suprema
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Defend minima compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.minima_pipeline);
            cpass.set_bind_group(0, &self.combine_bind_groups[0], &[]);
            cpass.set_bind_group(1, &self.combine_bind_groups[1], &[]);
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

    fn energies_equal(a: &[Energy], b: &[Energy]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let a_set: HashSet<Energy> = a.iter().copied().collect();
        for e in b {
            if !a_set.contains(e) {
                return false;
            }
        }
        true
    }

    fn process_results(&mut self, atk_visit_list: &mut Vec<u32>, game: &mut EnergyGame) {
        let minima_data = self.minima_staging_buf.slice(..).get_mapped_range();
        let minima: &[u64] = bytemuck::cast_slice(&minima_data);

        print!("Defend Minima:    ");
        for minima_chunk in minima {
            print!("{:064b} ", minima_chunk.reverse_bits());
        }
        println!("");

        let sup_data = self.sup_staging_buf.slice(..).get_mapped_range();
        let suprema: Vec<Energy> = bytemuck::cast_slice(&sup_data).to_vec();

        println!("Defend Suprema: {:?}", suprema);

        const MINIMA_SIZE: usize = 64;

        for node_window in self.node_offsets.windows(2) {
            let cur = node_window[0];
            let next_offset = node_window[1].sup_offset;
            let prev = &game.energies[cur.node as usize];

            // Maybe it is faster to first compare the number of entries by counting 1's in minima
            let new_energies: Vec<Energy> = suprema.iter()
                .enumerate()
                .take(next_offset as usize)
                .skip(cur.sup_offset as usize)
                // Read the corresponding minima flag
                .filter(|(i, _)| minima[i / MINIMA_SIZE] & (1 << i % MINIMA_SIZE) != 0)
                .map(|(_, e)| *e)
                .collect();

            if !Self::energies_equal(&new_energies, prev)  {
                // Write new, filtered energies
                game.energies[cur.node as usize] = new_energies;

                // Winning budgets have improved, check predecessors in next iteration
                for &pre in &game.graph.reverse[cur.node as usize] {
                    //TODO: Prune duplicate entries from both attack and defend shader
                    if game.graph.attacker_pos[pre as usize] {
                        atk_visit_list.push(pre);
                    } else {
                        self.visit_list.push(pre);
                    }
                }
            }
        }

        println!("Defend visit list: {:?}", self.visit_list);

        // Unmap buffers
        drop(minima_data);
        self.minima_staging_buf.unmap();
        drop(sup_data);
        self.sup_staging_buf.unmap();
    }
}

struct AttackShader {
    gpu: Rc<GPUCommon>,
    visit_list: Vec<u32>,
    energies: Vec<Energy>,
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
        game: &EnergyGame,
        graph_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<AttackShader> {
        let mut visit_list: Vec<u32> = Vec::new();
        for &v in &game.to_reach {
            // Start with parent nodes of final points
            for &w in &game.graph.reverse[v as usize] {
                if game.graph.attacker_pos[w as usize] {
                    visit_list.push(w);
                }
            }
        }

        let (node_offsets, successor_offsets, energies) = Self::collect_data(&visit_list, game)?;

        let node_offsets_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Node offsets storage buffer", Self::name())),
            contents: bytemuck::cast_slice(&node_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let energies_buf = Self::get_energies_buf(&gpu.device, &energies);
        let energies_staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Energies staging buffer", Self::name())),
            size: energies_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let successor_offsets_buf = Self::get_successor_offsets_buf(&gpu.device, &successor_offsets);
        let (minima_buf, minima_staging_buf) = Self::get_minima_buf(&gpu.device,
            Self::minima_size(energies.len()));

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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("attack.wgsl"))),
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

        Ok(AttackShader {
            gpu,
            visit_list,
            energies,
            energies_buf,
            energies_staging_buf,
            node_offsets,
            node_offsets_buf,
            successor_offsets_buf,
            minima_buf,
            minima_staging_buf,

            bind_group,
            bind_group_layout,
            pipeline,
            minima_pipeline,
        })
    }

    fn update(&mut self,
        game: &EnergyGame,
    ) -> Result<()> {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        let (node_offsets, successor_offsets, energies) = Self::collect_data(&self.visit_list, game)?;

        if !buffer_fits(&energies, &self.energies_buf) {
            self.energies_buf = Self::get_energies_buf(&device, &energies);
            self.energies_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Energies staging buffer", Self::name())),
                size: self.energies_buf.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        } else {
            queue.write_buffer(&self.energies_buf, 0, bytemuck::cast_slice(&energies));
        }
        self.energies = energies;

        // Make sure the node offsets buffer always has exactly the right size
        if node_offsets.len() * std::mem::size_of::<NodeOffsetAtk>() != self.node_offsets_buf.size() as usize {
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
            self.successor_offsets_buf = Self::get_successor_offsets_buf(&device, &successor_offsets);
        } else {
            queue.write_buffer(&self.successor_offsets_buf, 0, bytemuck::cast_slice(&successor_offsets));
        }

        let minima_capacity = Self::minima_size(self.energies.len());
        if minima_capacity * std::mem::size_of::<u32>() < self.minima_buf.size() as usize {
            (self.minima_buf, self.minima_staging_buf) = Self::get_minima_buf(&device, minima_capacity);
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
        Ok(())
    }

    fn collect_data(
        visit_list: &[u32],
        game: &EnergyGame
    ) -> Result<(Vec<NodeOffsetAtk>, Vec<u32>, Vec<Energy>)> {
        let mut energies = Vec::new();
        let mut successor_offsets = Vec::new();
        let mut successor_offsets_count = 0;
        let mut node_offsets = Vec::new();
        let mut count = 0;

        for node in visit_list {
            node_offsets.push(NodeOffsetAtk {
                node: *node,
                offset: count,
                successor_offsets_idx: successor_offsets_count,
            });
            for &successor in &game.graph.adj[*node as usize] {
                let successor_energies = &game.energies[successor as usize];
                energies.extend(successor_energies);
                successor_offsets.push(count);
                count = count.checked_add(successor_energies.len() as u32)
                    .ok_or(Error::Overflow)?;
            }
            // Last "successor" is always the own node in the attack case
            successor_offsets.push(count);
            let own_energies = &game.energies[*node as usize];
            energies.extend(own_energies);
            // Add number of successors + 1 for own node
            successor_offsets_count = successor_offsets_count.checked_add(
                game.graph.adj[*node as usize].len() as u32 + 1)
                .ok_or(Error::Overflow)?;
            count = count.checked_add(own_energies.len() as u32).ok_or(Error::Overflow)?;
        }
        successor_offsets.push(count);
        // Last offset does not correspond to another starting node, mark with u32::MAX
        node_offsets.push(NodeOffsetAtk {
            node: u32::MAX,
            offset: count,
            successor_offsets_idx: successor_offsets_count,
        });
        println!("Attack data: {:?}\n{:?}\n{:?}", node_offsets, successor_offsets, energies);
        Ok((node_offsets, successor_offsets, energies))
    }

    fn compute_pass(&self, encoder: &mut wgpu::CommandEncoder, graph_bind_group: &wgpu::BindGroup) {
        let n_energies = self.energies.len() as u32;
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
        }
        { // Compute pass for minimizing updated energies
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Attack minima compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.minima_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_bind_group(1, graph_bind_group, &[]);
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

    fn process_results(&mut self, def_visit_list: &mut Vec<u32>, game: &mut EnergyGame) {
        let minima_data = self.minima_staging_buf.slice(..).get_mapped_range();
        let minima: &[u64] = bytemuck::cast_slice(&minima_data);

        print!("Attack Minima:    ");
        for minima_chunk in minima {
            print!("{:064b} ", minima_chunk.reverse_bits());
        }
        println!("");

        let energies_data = self.energies_staging_buf.slice(..).get_mapped_range();
        let energies: Vec<Energy> = bytemuck::cast_slice(&energies_data).to_vec();

        println!("Attack Energies: {:?}", energies);

        const MINIMA_SIZE: u32 = 64;

        for node_window in self.node_offsets.windows(2) {
            let cur = node_window[0];
            let next_offset = node_window[1].offset;
            let width = next_offset - cur.offset;
            let n_prev = game.energies[cur.node as usize].len() as u32;
            let n_new = width - n_prev;
            assert!(n_prev <= width);

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
            // The previous energies at the end are all still present.
            //TODO: Rethink if this check is still needed.
            if !changed && n_prev > 0 {
                for minima_idx in ((cur.offset + n_new) / MINIMA_SIZE) ..= ((next_offset - 1) / MINIMA_SIZE) {
                    let shift_start = (cur.offset + n_new).max(minima_idx * MINIMA_SIZE) % MINIMA_SIZE;
                    let pad_right = (next_offset - 1).min((minima_idx + 1) * MINIMA_SIZE - 1) % MINIMA_SIZE;
                    let shift_end = MINIMA_SIZE - 1 - pad_right;
                    let mask = (u64::MAX << (shift_start + shift_end)) >> shift_end;

                    let minima_chunk = minima[minima_idx as usize];
                    if minima_chunk | !mask != u64::MAX {
                        changed = true;
                        break;
                    }
                }
            }

            if changed {
                // Write new, filtered energies
                game.energies[cur.node as usize] = energies.iter()
                    .enumerate()
                    .take(next_offset as usize)
                    .skip(cur.offset as usize)
                    // Read the corresponding minima flag
                    .filter(|(i, _)| minima[i / MINIMA_SIZE as usize] & (1 << i % MINIMA_SIZE as usize) > 0)
                    .map(|(_, e)| *e)
                    .collect();

                // Winning budgets have improved, check predecessors in next iteration
                for &pre in &game.graph.reverse[cur.node as usize] {
                    if game.graph.attacker_pos[pre as usize] {
                        self.visit_list.push(pre);
                    } else {
                        def_visit_list.push(pre);
                    }
                }
            }
        }

        println!("Attack visit list: {:?}", self.visit_list);

        // Unmap buffers
        drop(minima_data);
        self.minima_staging_buf.unmap();
        drop(energies_data);
        self.energies_staging_buf.unmap();
    }
}


// Common handles and data for managing the GPU device
#[derive(Debug)]
struct GPUCommon {
    device: Device,
    queue: Queue,
}

impl GPUCommon {
    async fn new() -> Result<GPUCommon> {
        let (device, queue) = Self::get_device().await?;

        Ok(GPUCommon {
            device,
            queue,
        })
    }

    async fn get_device() -> Result<(Device, Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or(Error::NoAdapter)?;

        Ok(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await?)
    }
}


pub struct GPURunner<'a> {
    game: &'a mut EnergyGame,

    gpu: Rc<GPUCommon>,
    _graph_bufs: (Buffer, Buffer, Buffer),
    graph_bind_group: wgpu::BindGroup,

    atk_shader: AttackShader,
    def_shader: DefendShader,
}

impl<'a> GPURunner<'a> {

    pub async fn with_game(game: &'a mut EnergyGame) -> Result<GPURunner<'a>> {
        let gpu_common = Rc::new(GPUCommon::new().await?);
        let graph_buffers = Self::graph_buffers(&game.graph, &gpu_common.device);
        let graph_bind_group_layout = Self::graph_bind_group_layout(&gpu_common.device);
        let graph_bind_group = Self::graph_bind_group(
            &graph_bind_group_layout, &gpu_common.device, &graph_buffers);

        let atk_shader = AttackShader::new(Rc::clone(&gpu_common), &game, &graph_bind_group_layout)?;
        let def_shader = DefendShader::new(Rc::clone(&gpu_common), &game, &graph_bind_group_layout)?;

        Ok(GPURunner {
            game,
            gpu: gpu_common,
            _graph_bufs: graph_buffers,
            graph_bind_group,

            atk_shader,
            def_shader,
        })
    }

    fn graph_buffers(graph: &GameGraph, device: &Device) -> (Buffer, Buffer, Buffer) {
        let (c, r, w) = graph.csr();
        let column_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input graph column indices storage bufer"),
            contents: bytemuck::cast_slice(&c),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let row_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input graph row offsets storage buffer"),
            contents: bytemuck::cast_slice(&r),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input graph edge weights storage buffer"),
            contents: bytemuck::cast_slice(&w),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        (column_indices_buffer, row_offsets_buffer, weights_buffer)
    }

    fn graph_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Input graph bind group layout"),
            entries: &[
                bgl_entry(0, true), // graph column indices
                bgl_entry(1, true), // graph row offsets
                bgl_entry(2, true), // graph edge weights
            ],
        })
    }

    fn graph_bind_group(layout: &wgpu::BindGroupLayout, device: &Device, buffers: &(Buffer, Buffer, Buffer))
    -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Input graph bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { // graph column indices
                    binding: 0,
                    resource: buffers.0.as_entire_binding(),
                },
                wgpu::BindGroupEntry { // graph row offsets
                    binding: 1,
                    resource: buffers.1.as_entire_binding(),
                },
                wgpu::BindGroupEntry { // graph edge weights
                    binding: 2,
                    resource: buffers.2.as_entire_binding(),
                },
            ],
        })
    }

    pub async fn execute_gpu(&mut self) -> Result<()> {
        println!("Attack visit list: {:?}", self.atk_shader.visit_list);
        println!("Defend visit list: {:?}", self.def_shader.visit_list);
        loop {
            let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Algorithm iteration encoder")
            });
            self.atk_shader.compute_pass(&mut encoder, &self.graph_bind_group);
            self.def_shader.compute_pass(&mut encoder, &self.graph_bind_group);

            // Submit command encoder for processing by GPU
            self.gpu.queue.submit(Some(encoder.finish()));

            const MAPS: usize = 4;
            let (sender, receiver) = channel(MAPS);
            self.atk_shader.map_buffers(&sender);
            self.def_shader.map_buffers(&sender);

            // Reset visit lists
            self.def_shader.visit_list.clear();
            self.atk_shader.visit_list.clear();

            // Wait for the GPU to finish work
            self.gpu.device.poll(wgpu::Maintain::Wait);

            for _ in 0..MAPS {
                receiver.receive().await.expect("Channel should not be closed")?;
            }

            self.atk_shader.process_results(&mut self.def_shader.visit_list, &mut self.game);
            self.def_shader.process_results(&mut self.atk_shader.visit_list, &mut self.game);

            if self.def_shader.visit_list.is_empty() && self.atk_shader.visit_list.is_empty() {
                // Nothing was updated, we are done.
                //return Ok(&self.game.energies);
                return Ok(());
            }

            self.atk_shader.update(&self.game)?;
            self.def_shader.update(&self.game)?;
        }
    }
}


// Convenience function for creating bind group layout entries
#[inline]
fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// Return true if buf is large enough to contain vec
fn buffer_fits<T>(vec: &Vec<T>, buf: &Buffer) -> bool {
    vec.len() * std::mem::size_of::<T>() <= buf.size() as usize
}

#[derive(Debug)]
pub enum Error {
    NoAdapter,
    NoDevice(wgpu::RequestDeviceError),
    Overflow,
    BufferMap(wgpu::BufferAsyncError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAdapter => write!(f, "could not get GPU adapter"),
            Error::NoDevice(source) => write!(f, "could not get GPU device: {}", source),
            Error::Overflow => write!(f, "overflow occured during calculation, input size too large"),
            Error::BufferMap(source) => source.fmt(f),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::NoDevice(source) => Some(source),
            Error::BufferMap(source) => Some(source),
            _ => None,
        }
    }
}

impl From<wgpu::RequestDeviceError> for Error {
    fn from(err: wgpu::RequestDeviceError) -> Error {
        Error::NoDevice(err)
    }
}

impl From<wgpu::BufferAsyncError> for Error {
    fn from(err: wgpu::BufferAsyncError) -> Error {
        Error::BufferMap(err)
    }
}

pub type Result<T> = result::Result<T, Error>;


#[cfg(test)]
mod tests;
