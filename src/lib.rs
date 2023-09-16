use std::borrow::Cow;
use std::iter;

use wgpu::{Buffer, Device, Queue};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;
const INF: u32 = 1 << 31;

#[derive(Debug, Clone)]
pub struct GameGraph {
    pub n_vertices: u32,
    pub adj: Vec<Vec<u32>>,
    pub reverse: Vec<Vec<u32>>,
    pub attacker_pos: Vec<bool>,
}

impl GameGraph {
    pub fn new(n_vertices: u32, edges: &[(u32, u32)], attacker_pos: &[bool]) -> Self {
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

pub struct EnergyGame {
    pub graph: GameGraph,
}

impl EnergyGame {

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

struct ShaderObjects {
    visit_list: Vec<u32>,
    visit_capacity: usize,
    visit_buf: Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

pub struct GPURunner<'a> {
    game: &'a mut EnergyGame,
    device: Device,
    queue: Queue,
    prev_output: Vec<u32>,

    output_buf: Buffer,
    staging_buf: Buffer,
    main_bind_group_layout: wgpu::BindGroupLayout,

    _graph_column_indices_buf: Buffer,
    _graph_row_offsets_buf: Buffer,
    _graph_attacker_pos_buf: Buffer,
    _graph_bind_group_layout: wgpu::BindGroupLayout,
    graph_bind_group: wgpu::BindGroup,

    def_shader: ShaderObjects,
    atk_shader: ShaderObjects,
}

impl<'a> GPURunner<'a> {

    pub async fn with_game(game: &'a mut EnergyGame) -> GPURunner<'a> {
        let (device, queue) = Self::get_device().await;

        let prev_output = vec![INF; game.graph.n_vertices as usize];
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

        let main_bind_group_layout = Self::main_bind_group_layout(&device);

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

        let def_shader = Self::defense_shader(&device, &game, &main_bind_group_layout, &shader, &pipeline_layout, &output_buf);
        let atk_shader = Self::attack_shader(&device, &main_bind_group_layout, &shader, &pipeline_layout, &output_buf);

        GPURunner {
            game,
            device,
            queue,

            prev_output,

            output_buf,
            staging_buf,
            main_bind_group_layout,

            _graph_column_indices_buf: graph_buffers.0,
            _graph_row_offsets_buf: graph_buffers.1,
            _graph_attacker_pos_buf: graph_buffers.2,
            _graph_bind_group_layout: graph_bind_group_layout,
            graph_bind_group,

            def_shader,
            atk_shader,
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

    fn attack_shader(
        device: &Device,
        main_bind_group_layout: &wgpu::BindGroupLayout,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        output_buf: &Buffer,
    ) -> ShaderObjects {
        // Initialize with empty attack visit list
        let visit_list = Vec::new();
        let visit_capacity = 8;  // Buffer size cannot be zero
        let visit_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("List of attack nodes to visit storage buffer"),
            size: (visit_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main attack shader bind group"),
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

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Attack compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "attack",
        });

        ShaderObjects {
            visit_list,
            visit_capacity,
            visit_buf,
            bind_group,
            pipeline,
        }
    }

    fn defense_shader(
        device: &Device,
        game: &EnergyGame,
        main_bind_group_layout: &wgpu::BindGroupLayout,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        output_buf: &Buffer,
    ) -> ShaderObjects {
        let mut visit_list = Vec::new();
        for i in 0..game.graph.n_vertices {
            // Initialize with all defender positions that have no outgoing edges
            if !game.graph.attacker_pos[i as usize] && game.graph.adj[i as usize].is_empty() {
                visit_list.push(i);
            }
        }

        let visit_capacity = visit_list.len();
        let visit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("List of defense nodes to visit storage buffer"),
            contents: bytemuck::cast_slice(&visit_list),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Defense compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "defend",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main defense shader bind group"),
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

        ShaderObjects {
            visit_list,
            visit_capacity,
            visit_buf,
            bind_group,
            pipeline,
        }
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

    pub async fn execute_gpu(&mut self) -> Result<Vec<Option<u32>>, String> {
        loop {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Algorithm iteration encoder")
            });
            for shader in [&self.def_shader, &self.atk_shader] {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cpass.set_pipeline(&shader.pipeline);
                cpass.set_bind_group(0, &shader.bind_group, &[]);
                cpass.set_bind_group(1, &self.graph_bind_group, &[]);

                // Ceil( n_visit / WORKGROUP_SIZE )
                let n_visit = shader.visit_list.len() as u32;
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


            self.def_shader.visit_list.clear();
            self.atk_shader.visit_list.clear();
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
                        for &w in &self.game.graph.reverse[v] {
                            if self.game.graph.attacker_pos[w as usize] {
                                self.atk_shader.visit_list.push(w);
                            } else {
                                self.def_shader.visit_list.push(w);
                            }
                        }
                    }
                }
                if self.def_shader.visit_list.is_empty() && self.atk_shader.visit_list.is_empty() {
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


#[cfg(test)]
mod tests;
