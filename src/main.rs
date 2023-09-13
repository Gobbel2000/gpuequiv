use std::borrow::Cow;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;
const INF: u32 = 1 << 31;

#[derive(Debug, Clone)]
struct GameGraph {
    n_vertices: usize,
    adj_matrix: Vec<Vec<bool>>,
    attacker_pos: Vec<bool>,
}

impl GameGraph {
    fn new(n_vertices: usize, edges: &[(usize, usize)], attacker_pos: &[bool]) -> Self {
        let mut adj_matrix = vec![vec![false; n_vertices]; n_vertices];
        for (from, to) in edges {
            adj_matrix[*from][*to] = true;
        }

        Self {
            n_vertices,
            adj_matrix,
            attacker_pos: attacker_pos.to_vec(),
        }
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
    // Boolean array to indicate the nodes that should be reached in the reachability game
    // Each bool is saved as u32 so it can be copied to the GPU
    let to_reach: Vec<u32> = graph.adj_matrix.iter()
        .zip(&graph.attacker_pos)
        // 1 if no outgoing edges and defender position, 0 otherwise
        .map(|(adj, atk)| u32::from(!atk && !adj.iter().any(|e| *e)))
        .collect();

    // BUFFERS

    // Copy the matrix into a flat array for the buffer
    // For now, bools are stored as u32, because the WGSL bool type cannot be used in storage
    // buffers (bool is not host-shareable).
    let flat_matrix: Vec<u32> = graph.adj_matrix
        .iter()
        .flatten()
        .map(|b| u32::from(*b))
        .collect();
    let adj_matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input graph adjacency matrix storage bufer"),
        contents: bytemuck::cast_slice(&flat_matrix),
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

    let output_size = (graph.n_vertices * std::mem::size_of::<u32>()) as u64;

    // Indicates whether each node should be visited in this iteration
    let visit_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Visit node flags storage buffer"),
        contents: bytemuck::cast_slice(&to_reach),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
    });

    let init_n_steps = vec![INF; graph.n_vertices];
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
            wgpu::BindGroupLayoutEntry {  // adj_matrix
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // attacker_pos
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // visit_buffer
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // output_buffer
                binding: 3,
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Main bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: adj_matrix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: attacker_pos_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: visit_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
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
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });


    for _ in 0..graph.n_vertices {
        // Program pipeline
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Algorithm iteration encoder")
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            // Ceil( n / WORKGROUP_SIZE )
            let n = graph.n_vertices as u32;
            let workgroup_count = if n % WORKGROUP_SIZE > 0 {
                n / WORKGROUP_SIZE + 1
            } else {
                n / WORKGROUP_SIZE
            };
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Submit command encoder for processing by GPU
        queue.submit(Some(encoder.finish()));
        // Wait for the GPU to finish work
        device.poll(wgpu::Maintain::Wait);
        
        //TODO: Copy and read visit_buffer and break loop if no more nodes need visiting
    }

    // Another command encoder for retrieving the result
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Result read encoder")
    });
    // Copy output to staging buffer
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    if let Some(Ok(())) = receiver.receive().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        result.into_iter()
            // High values mean no endpoint could be reached from this node
            .map(|n| if n >= INF { None } else { Some(n) })
            .collect()
    } else {
        panic!("failed to run compute on gpu!")
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
