use std::borrow::Cow;
use std::iter;

use wgpu::{Buffer, Device, Queue};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;
const INF: u32 = 1 << 31;

#[derive(Debug, Clone)]
struct GameGraph {
    n_vertices: u32,
    adj: Vec<Vec<u32>>,
    reverse: Vec<Vec<u32>>,
    attacker_pos: Vec<bool>,
}

impl GameGraph {
    fn new(n_vertices: u32, edges: &[(u32, u32)], attacker_pos: &[bool]) -> Self {
        let mut adj = vec![vec![]; n_vertices as usize];
        let mut reverse = vec![vec![]; n_vertices as usize];
        for (from, to) in edges {
            adj[*from as usize].push(*to);
            reverse[*to as usize].push(*from);
        }

        Self {
            n_vertices,
            adj,
            reverse,
            attacker_pos: attacker_pos.to_vec(),
        }
    }

    fn csr(&self) -> (Vec<u32>, Vec<u32>) {
        let column_indices = self.adj.iter()
            .flatten()
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
        (column_indices, row_offsets)
    }
}

fn example_graph() -> GameGraph {
    let attacker_pos: Vec<bool> = (0..18)
        .map(|i| [0, 2, 4, 6, 9, 11, 12, 14, 17].contains(&i))
        .collect();
    GameGraph::new(
        18,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (6, 8),
            (8, 5),
            (1, 9),
            (9, 10),
            (10, 11),
            (11, 3),
            (10, 12),
            (12, 10),
            (0, 13),
            (13, 14),
            (14, 15),
            (14, 16),
            (16, 17),
        ],
        &attacker_pos,
    )
}

struct EnergyGame {
    graph: GameGraph,
}

impl EnergyGame {
    fn example_graph() -> Self {
        Self {
            graph: example_graph(),
        }
    }

    async fn get_gpu_runner(&mut self) -> GPURunner {
        GPURunner::with_game(self).await
    }

    async fn run(&mut self) {
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

struct GPURunner<'a> {
    game: &'a mut EnergyGame,
    device: Device,
    queue: Queue,

    to_visit: Vec<u32>,
    visit_capacity: usize,
    prev_output: Vec<u32>,

    visit_buf: Buffer,
    output_buf: Buffer,
    staging_buf: Buffer,
    main_bind_group_layout: wgpu::BindGroupLayout,
    main_bind_group: wgpu::BindGroup,

    _graph_column_indices_buf: Buffer,
    _graph_row_offsets_buf: Buffer,
    _graph_attacker_pos_buf: Buffer,
    _graph_bind_group_layout: wgpu::BindGroupLayout,
    graph_bind_group: wgpu::BindGroup,

    pipeline: wgpu::ComputePipeline,
}

impl<'a> GPURunner<'a> {

    async fn with_game(game: &'a mut EnergyGame) -> GPURunner<'a> {
        let (device, queue) = Self::get_device().await;

        let graph = &game.graph;
        let mut to_visit = Vec::new();
        for i in 0..graph.n_vertices {
            // Initialize with all defender positions that have no outgoing edges
            if !graph.attacker_pos[i as usize] && graph.adj[i as usize].is_empty() {
                to_visit.push(i);
            }
        }

        let visit_capacity = to_visit.len();
        let visit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("List of nodes to visit storage buffer"),
            contents: bytemuck::cast_slice(&to_visit),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let prev_output = vec![INF; graph.n_vertices as usize];
        let output_size = (graph.n_vertices as usize * std::mem::size_of::<u32>()) as u64;
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

        let main_bind_group_layout = Self::main_bind_group_layout(&device);
        let main_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main bind group"),
            layout: &main_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: visit_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let graph_buffers = Self::graph_buffers(&game.graph, &device);
        let graph_bind_group_layout = Self::graph_bind_group_layout(&device);
        let graph_bind_group = Self::graph_bind_group(&graph_bind_group_layout, &device, &graph_buffers);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute shader module"),
            // Shader source embedded in binary
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline layout"),
            bind_group_layouts: &[&main_bind_group_layout, &graph_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        GPURunner {
            game,
            device,
            queue,

            to_visit,
            visit_capacity,
            prev_output,

            visit_buf,
            output_buf,
            staging_buf,
            main_bind_group_layout,
            main_bind_group,

            _graph_column_indices_buf: graph_buffers.0,
            _graph_row_offsets_buf: graph_buffers.1,
            _graph_attacker_pos_buf: graph_buffers.2,
            _graph_bind_group_layout: graph_bind_group_layout,
            graph_bind_group,

            pipeline,
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

    fn main_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main bind group layout"),
            entries: &[
                bgl_entry(0, true), // visit_buffer
                bgl_entry(1, false), // output_buffer
            ],
        })
    }

    fn graph_buffers(graph: &GameGraph, device: &Device) -> (Buffer, Buffer, Buffer) {
        let (c, r) = graph.csr();
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

        let attacker_pos_u32: Vec<u32>  = graph.attacker_pos
           .iter()
           .map(|b| u32::from(*b))
           .collect();
        let attacker_pos_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input graph attacker positions storage buffer"),
            contents: bytemuck::cast_slice(&attacker_pos_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        (column_indices_buffer, row_offsets_buffer, attacker_pos_buffer)
    }

    fn graph_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Input graph bind group layout"),
            entries: &[
                bgl_entry(0, true), // graph column indices
                bgl_entry(1, true), // graph row offsets
                bgl_entry(2, true), // graph attacker pos
            ],
        })
    }

    fn graph_bind_group(layout: &wgpu::BindGroupLayout, device: &Device, buffers: &(Buffer, Buffer, Buffer))
    -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Input graph bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.2.as_entire_binding(),
                },
            ],
        })
    }

    // Update the storage buffer, and resize that buffer if necessary, to accommodate new to_visit
    fn set_visit_list(&mut self) {
        // Resize the buffer if necessary to hold all nodes that need to be visited
        if self.visit_capacity < self.to_visit.len() {
            self.visit_capacity = self.to_visit.len();
            self.visit_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("List of nodes to visit storage buffer"),
                size: (self.visit_capacity * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.main_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Main bind group"),
                layout: &self.main_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.visit_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.output_buf.as_entire_binding(),
                    },
                ],
            });
        }

        // Schedule writing new visit list in buffer
        self.queue.write_buffer(&self.visit_buf, 0, bytemuck::cast_slice(&self.to_visit));
    }

    async fn execute_gpu(&mut self) -> Result<Vec<Option<u32>>, String> {
        loop {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Algorithm iteration encoder")
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &self.main_bind_group, &[]);
                cpass.set_bind_group(1, &self.graph_bind_group, &[]);

                // Ceil( n_visit / WORKGROUP_SIZE )
                let n_visit = self.to_visit.len() as u32;
                let workgroup_count = if n_visit % WORKGROUP_SIZE > 0 {
                    n_visit / WORKGROUP_SIZE + 1
                } else {
                    n_visit / WORKGROUP_SIZE
                };
                cpass.dispatch_workgroups(workgroup_count, 1, 1);
            }

            // Copy output to staging buffer
            encoder.copy_buffer_to_buffer(&self.output_buf, 0, &self.staging_buf, 0, self.output_buf.size());
            // Submit command encoder for processing by GPU
            self.queue.submit(Some(encoder.finish()));

            let buffer_slice = self.staging_buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());


            self.to_visit.clear();
            // Wait for the GPU to finish work
            self.device.poll(wgpu::Maintain::Wait);

            if let Some(Ok(())) = receiver.receive().await {
                let data = buffer_slice.get_mapped_range();
                let output: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

                drop(data);
                self.staging_buf.unmap();

                for (v, (new_step, prev_step)) in iter::zip(&output, &self.prev_output).enumerate() {
                    if new_step < prev_step {
                        // Distance was updated: Visit parent nodes next.
                        for w in &self.game.graph.reverse[v] {
                            self.to_visit.push(*w);
                        }
                    }
                }
                if self.to_visit.is_empty() {
                    // Nothing was updated, we are done.
                    return Ok(output.into_iter()
                        // High values mean no endpoint could be reached from this node
                        .map(|n| if n >= INF { None } else { Some(n) })
                        .collect())
                }
                self.set_visit_list();
                self.prev_output = output;
            } else {
                return Err("Failed to run compute on gpu!".to_string())
            }
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

async fn run() {
    let mut game = EnergyGame::example_graph();
    game.run().await;
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        simple_logger::init_with_level(log::Level::Warn).expect("Could not initialize logger");
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}
