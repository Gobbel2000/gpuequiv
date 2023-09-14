use std::borrow::Cow;
use std::iter;

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
                *state = *state + adj.len() as u32;
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

async fn run() {
    let graph = example_graph();
    let n_steps = execute_gpu(&graph).await;

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

async fn execute_gpu(graph: &GameGraph) -> Vec<Option<u32>> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("Could not get wgpu adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .expect("Could not get device");
    
    execute_gpu_inner(&device, &queue, graph).await
}

async fn execute_gpu_inner(device: &wgpu::Device, queue: &wgpu::Queue, graph: &GameGraph) -> Vec<Option<u32>> {
    let mut to_visit: Vec<u32> = Vec::new();
    for i in 0..graph.n_vertices {
        // Initialize with all defender positions that have no outgoing edges
        if !graph.attacker_pos[i as usize] && graph.adj[i as usize].is_empty() {
            to_visit.push(i);
        }
    }

    // BUFFERS

    // Copy the matrix into a flat array for the buffer
    // For now, bools are stored as u32, because the WGSL bool type cannot be used in storage
    // buffers (bool is not host-shareable).
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

    let output_size = (graph.n_vertices as usize * std::mem::size_of::<u32>()) as u64;

    let init_n_steps = vec![INF; graph.n_vertices as usize];
    // Output buffer that holds the number of steps for each node required to reach a wanted end point
    let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Shader output buffer"),
        contents: bytemuck::cast_slice(&init_n_steps),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Staging buffer for mapping the output buffer back to CPU memory for reading
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shader output staging buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });


    // BIND GROUP
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Main bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {  // visit_buffer
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // output_buffer
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let graph_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Input graph bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {  // graph column indices
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // graph row offsets
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // attacker_pos
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let graph_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Input graph bind group"),
        layout: &graph_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: column_indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: row_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: attacker_pos_buffer.as_entire_binding(),
            },
        ],
    });

    // PIPELINE
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute shader module"),
        // Shader source embedded in binary
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline layout"),
        bind_group_layouts: &[&bind_group_layout, &graph_bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let mut prev = init_n_steps;
    let mut visit_buffer_size = to_visit.len();
    let mut visit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("List of nodes to visit storage buffer"),
        size: (visit_buffer_size * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    loop {
        // Resize the buffer if necessary to hold all nodes that need to be visited
        if visit_buffer_size < to_visit.len() {
            visit_buffer_size = to_visit.len();
            visit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("List of nodes to visit storage buffer"),
                size: (visit_buffer_size * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&visit_buffer, 0, bytemuck::cast_slice(&to_visit));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: visit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Program pipeline
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Algorithm iteration encoder")
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_bind_group(1, &graph_bind_group, &[]);

            // Ceil( n_visit / WORKGROUP_SIZE )
            let n_visit = to_visit.len() as u32;
            let workgroup_count = if n_visit % WORKGROUP_SIZE > 0 {
                n_visit / WORKGROUP_SIZE + 1
            } else {
                n_visit / WORKGROUP_SIZE
            };
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        // Submit command encoder for processing by GPU
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        to_visit.clear();

        // Wait for the GPU to finish work
        device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let output: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            staging_buffer.unmap();

            for (v, (new_step, prev_step)) in iter::zip(&output, &prev).enumerate() {
                if new_step < prev_step {
                    // Distance was updated: Visit parent nodes next.
                    for w in &graph.reverse[v] {
                        to_visit.push(*w);
                    }
                }
            }
            if to_visit.is_empty() {
                // Nothing was updated, we are done.
                return output.into_iter()
                    // High values mean no endpoint could be reached from this node
                    .map(|n| if n >= INF { None } else { Some(n) })
                    .collect()
            }
            prev = output;
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
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
