use std::borrow::Cow;
use std::iter;

use wgpu::{Buffer, Device, Queue};
use wgpu::util::DeviceExt;

// Spawn 64 threads with each workgroup invocation
const WORKGROUP_SIZE: u32 = 64;
const INF: u32 = 1 << 31;

#[derive(Debug, Clone)]
pub struct GameGraph {
    pub n_vertices: u32,
    pub adj: Vec<Vec<u32>>,
    pub reverse: Vec<Vec<u32>>,
    pub weights: Vec<Vec<Energy>>,
    pub attacker_pos: Vec<bool>,
}

impl GameGraph {
    pub fn new(n_vertices: u32, edges: &[(u32, u32, Energy)], attacker_pos: &[bool]) -> Self {
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
            .map(|e| e.data)
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

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Energy {
    data: u32,
}

impl From<u32> for Energy {
    fn from(data: u32) -> Self {
        Energy { data }
    }
}

impl Default for Energy {
    fn default() -> Self {
        Energy::zero()
    }
}

impl Energy {
    fn zero() -> Self {
        Energy { data: 0 }
    }
}

// Enable bytemucking for filling buffers
unsafe impl bytemuck::Zeroable for Energy {}
unsafe impl bytemuck::Pod for Energy {}


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

    pub async fn get_gpu_runner(&mut self) -> GPURunner {
        GPURunner::with_game(self).await
    }

    pub async fn run(&mut self) {
        let mut runner = self.get_gpu_runner().await;
        let n_steps = runner.execute_gpu().await.unwrap();

        // Pretty-print output
        let mut out = String::new();
        for (i, step) in n_steps.iter().enumerate() {
            match step {
                Some(n) => out += &format!("\n{}:\t{}", i, n),
                None => out += &format!("\n{}:\t{}", i, '-'),
            }
        }
        log::warn!("{}", out);
    }

}

#[repr(C)]  // Needed for safely implementing Pod
#[derive(Clone, Copy, Debug)]
struct NodeOffset {
    node: u32,
    offset: u32,
}
// Enable bytemucking for filling buffers
unsafe impl bytemuck::Zeroable for NodeOffset {}
unsafe impl bytemuck::Pod for NodeOffset {}

struct ShaderObjects {
    visit_list: Vec<u32>,
    energies: Vec<Energy>,
    energies_buf: Buffer,
    energies_staging_buf: Buffer,
    node_offsets: Vec<NodeOffset>,
    node_offsets_buf: Buffer,
    successor_offsets_buf: Buffer,
    minima_buf: Buffer,
    minima_staging_buf: Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

impl ShaderObjects {
    fn per_player_shader(
        is_attack: bool,
        device: &Device,
        game: &EnergyGame,
        bind_group_layout: &wgpu::BindGroupLayout,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
    ) -> ShaderObjects {
        let visit_list: Vec<u32> = game.to_reach.iter()
            .filter(|&v| game.graph.attacker_pos[*v as usize] ^ !is_attack)
            .copied()
            .collect();

        let (node_offsets, successor_offsets, energies) = GPURunner::collect_data(&visit_list, game);

        let player = if is_attack { "Attack" } else { "Defense" };

        let energies_buf;
        if energies.is_empty() {
            // Initialize 0-filled buffers with some initial size.
            // Buffers with size 0 are not allowed.
            const INITIAL_ENERGIES_CAPACITY: usize = 64;
            energies_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Energies storage buffer", player)),
                size: (INITIAL_ENERGIES_CAPACITY * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        } else {
            energies_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Energies storage buffer", player)),
                contents: bytemuck::cast_slice(&energies),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            });
        }
        let energies_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Energies staging buffer", player)),
            size: energies_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let node_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Node offsets storage buffer", player)),
            contents: bytemuck::cast_slice(&node_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let successor_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Successor offsets storage buffer", player)),
            contents: bytemuck::cast_slice(&successor_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Output flags are bit-packed with 32 bools per u32. Round up to next multiple of 64.
        // minima_capacity is measured in number of u32's.
        let minima_capacity = (((energies.len() as i64 - 1) / 64 + 1) * 2) as usize;
        let minima_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Minimal energy flags storage buffer", player)),
            size: (minima_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // For reading minima_buf on the CPU
        let minima_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Minima output staging buffer", player)),
            size: minima_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Main shader bind group", player)),
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

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} Minima compute pipeline", player)),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "process_energies",
        });

        ShaderObjects {
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
            pipeline,
        }
    }

    fn update(&mut self,
        is_attack: bool,
        device: &Device,
        queue: &Queue,
        game: &EnergyGame,
        bind_group_layout: &wgpu::BindGroupLayout,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
    ) {
        let visit_list: Vec<u32> = game.to_reach.iter()
            .filter(|&v| game.graph.attacker_pos[*v as usize] ^ !is_attack)
            .copied()
            .collect();

        let (node_offsets, successor_offsets, energies) = GPURunner::collect_data(&visit_list, game);

        let player = if is_attack { "Attack" } else { "Defense" };

        self.energies = energies;
        if self.energies.len() > self.energies_buf.size() as usize {
            self.energies_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Energies storage buffer", player)),
                contents: bytemuck::cast_slice(&self.energies),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            });
            self.energies_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Energies staging buffer", player)),
                size: self.energies_buf.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        } else {
            queue.write_buffer(&self.energies_buf, 0, bytemuck::cast_slice(&self.energies));
        }

        self.node_offsets = node_offsets;
        if self.node_offsets.len() > self.node_offsets_buf.size() as usize {
            self.node_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Node offsets storage buffer", player)),
                contents: bytemuck::cast_slice(&self.node_offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&self.node_offsets_buf, 0, bytemuck::cast_slice(&self.node_offsets));
        }

        if successor_offsets.len() > self.successor_offsets_buf.size() as usize {
            self.successor_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Successor offsets storage buffer", player)),
                contents: bytemuck::cast_slice(&successor_offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&self.successor_offsets_buf, 0, bytemuck::cast_slice(&successor_offsets));
        }

        // Output flags are bit-packed with 32 bools per u32. Round up to next multiple of 64.
        // minima_capacity is measured in number of u32's.
        let minima_capacity = (((self.energies.len() as i64 - 1) / 64 + 1) * 2) as usize;
        if minima_capacity < self.minima_buf.size() as usize {
            self.minima_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Minimal energy flags storage buffer", player)),
                size: (minima_capacity * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            // For reading minima_buf on the CPU
            self.minima_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Minima output staging buffer", player)),
                size: self.minima_buf.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Main shader bind group", player)),
            layout: &bind_group_layout,
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

        self.pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} Minima compute pipeline", player)),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "process_energies",
        });
    }
}


pub struct GPURunner<'a> {
    game: &'a mut EnergyGame,
    device: Device,
    queue: Queue,
    prev_output: Vec<u32>,

    energy_bind_group_layout: wgpu::BindGroupLayout,

    _graph_bufs: (Buffer, Buffer, Buffer),
    _graph_bind_group_layout: wgpu::BindGroupLayout,
    graph_bind_group: wgpu::BindGroup,
    minima_shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,

    atk_shader: ShaderObjects,
    def_shader: ShaderObjects,
}

impl<'a> GPURunner<'a> {

    pub async fn with_game(game: &'a mut EnergyGame) -> GPURunner<'a> {
        let (device, queue) = Self::get_device().await;

        let prev_output = vec![INF; game.graph.n_vertices as usize];
        /*
        let output_size = (game.graph.n_vertices as usize * std::mem::size_of::<u32>()) as u64;
        // Output buffer that holds the number of steps for each node required to reach a wanted end point
        let output_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shader output buffer"),
            contents: bytemuck::cast_slice(&prev_output),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        // Staging buffer for mapping the output buffer back to CPU memory for reading
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shader output staging buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        */

        let energy_bind_group_layout = Self::energy_bind_group_layout(&device);

        let graph_buffers = Self::graph_buffers(&game.graph, &device);
        let graph_bind_group_layout = Self::graph_bind_group_layout(&device);
        let graph_bind_group = Self::graph_bind_group(&graph_bind_group_layout, &device, &graph_buffers);

        /*
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute shader module"),
            // Shader source embedded in binary
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });
        */
        let minima_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Minima shader module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("minima.wgsl"))),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline layout"),
            bind_group_layouts: &[&energy_bind_group_layout, &graph_bind_group_layout],
            push_constant_ranges: &[],
        });

        let atk_shader = ShaderObjects::per_player_shader(
            true,  &device, &game, &energy_bind_group_layout, &minima_shader, &pipeline_layout);
        let def_shader = ShaderObjects::per_player_shader(
            false, &device, &game, &energy_bind_group_layout, &minima_shader, &pipeline_layout);

        GPURunner {
            game,
            device,
            queue,

            prev_output,
            energy_bind_group_layout,
            _graph_bufs: graph_buffers,
            _graph_bind_group_layout: graph_bind_group_layout,
            graph_bind_group,
            minima_shader,
            pipeline_layout,

            atk_shader,
            def_shader,
        }
    }

    async fn get_device() -> (Device, Queue) {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("Could not get wgpu adapter");

        adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .expect("Could not get device")
    }

    /*
    fn main_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main bind group layout"),
            entries: &[
                bgl_entry(0, true), // visit_buffer
                bgl_entry(1, false), // output_buffer
            ],
        })
    }
    */

    fn energy_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Energy bind group layout"),
            entries: &[
                bgl_entry(0, false), // energies, writable
                bgl_entry(1, true),  // node offsets
                bgl_entry(2, true),  // successor offsets
                bgl_entry(3, false), // minima, writable
            ],
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

    /*
    // Update the storage buffer, and resize that buffer if necessary, to accommodate new to_visit
    fn set_visit_list(&mut self) {
        // Resize the buffer if necessary to hold all nodes that need to be visited
        for shader in [&mut self.def_shader, &mut self.atk_shader] {
            if shader.visit_capacity < shader.visit_list.len() {
                shader.visit_capacity = shader.visit_list.len();
                shader.visit_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("List of nodes to visit storage buffer"),
                    size: (shader.visit_capacity * std::mem::size_of::<u32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                shader.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Main bind group"),
                    layout: &self.main_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: shader.visit_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.output_buf.as_entire_binding(),
                        },
                    ],
                });
            }
            // Schedule writing new visit list in buffer
            self.queue.write_buffer(&shader.visit_buf, 0, bytemuck::cast_slice(&shader.visit_list));
        }
    }
    */

    fn prepare_buffers(&mut self) {
        self.atk_shader.update(
            true, &self.device, &self.queue, &self.game,
            &self.energy_bind_group_layout, &self.minima_shader, &self.pipeline_layout);
        self.atk_shader.update(
            false, &self.device, &self.queue, &self.game,
            &self.energy_bind_group_layout, &self.minima_shader, &self.pipeline_layout);
    }

    fn collect_data(visit_list: &[u32], game: &EnergyGame) -> (Vec<NodeOffset>, Vec<u32>, Vec<Energy>) {
        let mut energies = Vec::new();
        let mut successor_offsets = Vec::new();
        let mut node_offsets = Vec::new();
        let mut count: u32 = 0;
        for node in visit_list {
            node_offsets.push(NodeOffset { node: *node, offset: count });
            for (successor_idx, successor) in game.graph.adj[*node as usize].iter().enumerate() {
                //successor_offsets.push(count);
                let successor_energies = &game.energies[*successor as usize];
                energies.extend(successor_energies);
                //TODO: Reevaluate this mechanism of repeating the successor index
                successor_offsets.extend(iter::repeat(successor_idx as u32).take(successor_energies.len()));
                count += successor_energies.len() as u32;
            }
        }
        successor_offsets.push(count);
        // Last offset does not correspond to another starting node, mark with u32::MAX
        node_offsets.push(NodeOffset { node: u32::MAX, offset: count });
        (node_offsets, successor_offsets, energies)
    }

    pub async fn execute_gpu(&mut self) -> Result<Vec<Option<u32>>, String> {
        loop {
            println!("Visit list: {:?}", self.atk_shader.visit_list);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Algorithm iteration encoder")
            });
            for shader in [&self.atk_shader] { //TODO: add back &self.def_shader, 
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                    cpass.set_pipeline(&shader.pipeline);
                    cpass.set_bind_group(0, &shader.bind_group, &[]);
                    cpass.set_bind_group(1, &self.graph_bind_group, &[]);

                    // Ceil( n_visit / WORKGROUP_SIZE )
                    let n_energies = shader.energies.len() as u32;
                    let workgroup_count = if n_energies % WORKGROUP_SIZE > 0 {
                        n_energies / WORKGROUP_SIZE + 1
                    } else {
                        n_energies / WORKGROUP_SIZE
                    };
                    cpass.dispatch_workgroups(workgroup_count, 1, 1);
                }
                // Copy output to staging buffer
                encoder.copy_buffer_to_buffer(&shader.minima_buf, 0, &shader.minima_staging_buf, 0, shader.minima_buf.size());
                encoder.copy_buffer_to_buffer(&shader.energies_buf, 0, &shader.energies_staging_buf, 0, shader.energies_buf.size());
            }

            // Submit command encoder for processing by GPU
            self.queue.submit(Some(encoder.finish()));

            let minima_buffer_slice = self.atk_shader.minima_staging_buf.slice(..);  //TODO: Handle output of def_shader too
            let energies_buffer_slice = self.atk_shader.energies_staging_buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::channel(2);
            let sender2 = sender.clone();
            minima_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
                sender.try_send(v).unwrap();
            });
            energies_buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
                sender2.try_send(v).unwrap();
            });

            // Reset visit lists
            self.def_shader.visit_list.clear();
            self.atk_shader.visit_list.clear();
            // Wait for the GPU to finish work
            self.device.poll(wgpu::Maintain::Wait);

            let r1 = receiver.receive().await;
            let r2 = receiver.receive().await;

            if r1 != Some(Ok(())) || r2 != Some(Ok(())) {
                return Err("Error while mapping buffers".to_string());
            }

            let minima = minima_buffer_slice.get_mapped_range();
            let minima_bools: Vec<u64> = bytemuck::cast_slice(&minima).to_vec();
            println!("Minima: {:?}", minima_bools);
            let energies_data = energies_buffer_slice.get_mapped_range();
            let energies: Vec<u32> = bytemuck::cast_slice(&energies_data).to_vec();
            println!("Energies: {:?}", energies);

            const MINIMA_SIZE: u32 = 8;

            for node_window in self.atk_shader.node_offsets.windows(2) {
                let cur = node_window[0];
                let next_offset = node_window[1].offset;
                let width = next_offset - cur.offset;
                let n_prev = self.game.energies[cur.node as usize].len() as u32;
                let n_new = width - n_prev;
                assert!(n_prev <= width);

                let mut changed = false;
                // If energies didn't change, all newly compared energies will be filtered out
                for minima_idx in (cur.offset / MINIMA_SIZE) ..= ((cur.offset + n_new - 1) / MINIMA_SIZE) {
                    // What position in the minima u64 the node starts at
                    let shift_start = cur.offset.max(minima_idx * MINIMA_SIZE) % MINIMA_SIZE;
                    // ... and ends at
                    let shift_end = (cur.offset + n_new - 1).min((minima_idx + 1) * MINIMA_SIZE - 1) % MINIMA_SIZE;
                    let mask = (u64::MAX << (shift_start + shift_end)) >> shift_end;

                    let minima_chunk = minima_bools[minima_idx as usize];
                    if minima_chunk & mask != 0 {
                        changed = true;
                    }
                }
                // The previous energies at the end are all still present.
                //TODO: Rethink if this check is still needed.
                if !changed {
                    for minima_idx in ((cur.offset + n_new) / MINIMA_SIZE) ..= ((next_offset - 1) / MINIMA_SIZE) {
                        let shift_start = (cur.offset + n_new).max(minima_idx * MINIMA_SIZE) % MINIMA_SIZE;
                        let shift_end = (next_offset - 1).min((minima_idx + 1) * MINIMA_SIZE - 1) % MINIMA_SIZE;
                        let mask = (u64::MAX << (shift_start + shift_end)) >> shift_end;

                        let minima_chunk = minima_bools[minima_idx as usize];
                        if minima_chunk | !mask != u64::MAX {
                            changed = true;
                        }
                    }
                }

                if changed {
                    // Winning budgets have improved, check predecessors in next iteration
                    for &pre in &self.game.graph.reverse[cur.node as usize] {
                        if self.game.graph.attacker_pos[pre as usize] {
                            self.atk_shader.visit_list.push(pre);
                        } else {
                            self.def_shader.visit_list.push(pre);
                        }
                    }
                }
            }

            drop(minima);
            self.atk_shader.minima_staging_buf.unmap();
            drop(energies_data);
            self.atk_shader.energies_staging_buf.unmap(); //TODO: do def_shader too

            if self.def_shader.visit_list.is_empty() && self.atk_shader.visit_list.is_empty() {
                // Nothing was updated, we are done.
                return Ok(Vec::new()); //TODO: Return value
            }

            self.prepare_buffers();
            return Ok(Vec::new());
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


#[cfg(test)]
mod tests;
