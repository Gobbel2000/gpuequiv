use std::{mem, iter};
use std::borrow::Cow;
use std::rc::Rc;

use wgpu::util::DeviceExt;
use wgpu::{BufferUsages, Buffer};

use crate::utils::{GPUCommon, bgl_entry, GPUGraph};
use crate::error::Result;

pub struct TransitionSystem {
    pub adj: Vec<Vec<u32>>,
    // Transition labels encoded as i32:
    // 0 => τ
    // k => Channel index k, k ∈ ℕ
    // -k => Co-Action of k, k ∈ ℕ
    //
    // Actual names (Strings) should be stored in a separate list
    pub labels: Vec<Vec<i32>>,
}

impl TransitionSystem {
    pub fn new(n_vertices: u32, edges: Vec<(u32, u32, i32)>) -> Self {
        let mut adj = vec![vec![]; n_vertices as usize];
        let mut labels = vec![vec![]; n_vertices as usize];
        for (from, to, label) in edges {
            adj[from as usize].push(to);
            labels[from as usize].push(label);
        }
        TransitionSystem {
            adj,
            labels,
        }
    }

    pub fn n_vertices(&self) -> u32 {
        self.adj.len() as u32
    }
}

impl GPUGraph for TransitionSystem {
    type Weight = i32;

    fn csr(&self) -> (Vec<u32>, Vec<u32>, Vec<Self::Weight>) {
        let column_indices = self.adj.iter()
            .flatten()
            .copied()
            .collect();
        let labels = self.labels.iter()
            .flatten()
            .copied()
            .collect();
        let row_offsets = iter::once(0).chain(
            self.adj.iter()
            .scan(0, |state, adj| {
                *state += adj.len() as u32;
                Some(*state)
            }))
            .collect();
        (column_indices, row_offsets, labels)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct LinkedList {
    data: [u32; 4],
    len: u32,
    next: u32,
}

// Enable bytemucking for filling buffers
unsafe impl bytemuck::Zeroable for LinkedList {}
unsafe impl bytemuck::Pod for LinkedList {}

#[repr(C)]
#[allow(non_snake_case)]
#[derive(Debug, Copy, Clone)]
struct Position {
    p: u32,
    Q: LinkedList,
}

unsafe impl bytemuck::Zeroable for Position {}
unsafe impl bytemuck::Pod for Position {}

#[repr(C)]
#[allow(non_snake_case)]
#[derive(Debug, Copy, Clone)]
struct ConjunctionPosition {
    p: u32,
    Q: LinkedList,
    Qx: LinkedList,
}

unsafe impl bytemuck::Zeroable for ConjunctionPosition {}
unsafe impl bytemuck::Pod for ConjunctionPosition {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Metadata {
    heap_top: u32,
    heap_oom: u32,
}

unsafe impl bytemuck::Zeroable for Metadata {}
unsafe impl bytemuck::Pod for Metadata {}


struct ClauseShader {
    gpu: Rc<GPUCommon>,
    positions: Vec<Position>,
    buf_a: Buffer,
    buf_b: Buffer,
    staging_buf: Buffer,

    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl ClauseShader {
    fn new(gpu: Rc<GPUCommon>) -> Self {
        let pos = vec![
            Position {
                p: 4,
                Q: LinkedList { data: [3, 0, 0, 0], len: 1, next: 0 },
            },
            Position {
                p: 7,
                Q: LinkedList { data: [9, 2, 0, 0], len: 2, next: 0 },
            },
        ];

        let buf_a = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clause buffer A"),
            contents: bytemuck::cast_slice(&pos),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let buf_b = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Clause buffer B"),
            size: 280,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Clause staging buffer"),
            size: 280,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Clause bind group layout"),
            entries: &[
                bgl_entry(0, true), // Input, readonly
                bgl_entry(1, false), // Output, writable
            ],
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clause bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Clause pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clause shader module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("clause.wgsl"))),
        });

        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clause compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "process_clauses",
        });

        ClauseShader {
            gpu,
            positions: pos,
            buf_a,
            buf_b,
            staging_buf,
            bind_group,
            bind_group_layout,
            pipeline,
        }
    }
}

struct ConjunctionShader {
    gpu: Rc<GPUCommon>,
    positions: Vec<ConjunctionPosition>,
    conjunction_buf: Buffer,
    out_buf: Buffer,
    staging_buf: Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl ConjunctionShader {
    fn new(gpu: Rc<GPUCommon>) -> Self {
        let positions = Vec::new();

        let conjunction_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Conjunction input storage buffer"),
            contents: bytemuck::cast_slice(&positions),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conjunction output storage buffer"),
            size: 1024, //TODO
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conjunction output read staging buffer"),
            size: out_buf.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let heap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conjunction output linked list heap storage buffer"),
            size: (1024 * mem::size_of::<LinkedList>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let heap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conjunction output linked list heap storage buffer"),
            size: (1024 * mem::size_of::<LinkedList>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let metadata_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Conjunction metadata storage buffer"),
            contents: bytemuck::bytes_of(&Metadata::default()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Conjunction bind group layout"),
            entries: &[
                bgl_entry(0, true), // Input, read only
                bgl_entry(1, false), // Ouput, writable
            ],
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Conjunction bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: conjunction_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Conjunction shader module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("clause.wgsl"))),
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Conjunction pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Conjunction compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "process_conjunctions",
        });

        ConjunctionShader {
            gpu,
            positions,
            conjunction_buf,
            out_buf,
            staging_buf,
            bind_group,
            bind_group_layout,
            pipeline,
        }
    }
}


struct ChallengeShader {
    gpu: Rc<GPUCommon>,
    positions: Vec<Position>,
    in_buf: Buffer,
    out_buf: Buffer,
    heap_buf: Buffer,
    metadata_buf: Buffer,
    staging_buf: Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    bind_group_layouts: [wgpu::BindGroupLayout; 2],
    pipeline: wgpu::ComputePipeline,
}

impl ChallengeShader {
    fn new(gpu: Rc<GPUCommon>, graph_layout: &wgpu::BindGroupLayout) -> Self {
        let positions = vec![
            Position {
                p: 1,
                Q: LinkedList { data: [0, 2, 0, 0], len: 2, next: 0 },
            },
        ];

        let in_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Challenge input storage buffer"),
            contents: bytemuck::cast_slice(&positions),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Challenge output storage buffer"),
            // Allocate size for 4 output positions per input position
            size: (positions.len() * 4 * mem::size_of::<ConjunctionPosition>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let heap_in_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conjunction output linked list heap storage buffer"),
            size: (16 * mem::size_of::<LinkedList>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let heap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conjunction output linked list heap storage buffer"),
            size: (16 * mem::size_of::<LinkedList>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let metadata_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Conjunction metadata storage buffer"),
            contents: bytemuck::bytes_of(&Metadata::default()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Challenge output read staging buffer"),
            size: out_buf.size() + heap_buf.size() + metadata_buf.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layouts = [
            gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Challenge bind group layout 0"),
                entries: &[
                    bgl_entry(0, true), // Input, read only
                    bgl_entry(1, false), // Ouput, writable
                    bgl_entry(2, true), // Input heap, read only
                    bgl_entry(3, false), // Output heap, writable
                ],
            }),
            gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Challenge bind group layout 1"),
                entries: &[
                    bgl_entry(0, false), // Metadata, writable
                ],
            }),
        ];

        let bind_groups = [
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Challenge bind group 0"),
                layout: &bind_group_layouts[0],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: heap_in_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: heap_buf.as_entire_binding(),
                    },
                ],
            }),
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Challenge bind group 1"),
                layout: &bind_group_layouts[1],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: metadata_buf.as_entire_binding(),
                    },
                ],
            }),
        ];

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Challenge shader module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("challenge.wgsl"))),
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Challenge pipeline layout"),
            bind_group_layouts: &[&bind_group_layouts[0], &bind_group_layouts[1], &graph_layout],
            push_constant_ranges: &[],
        });

        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Challenge compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "process_challenges",
        });

        ChallengeShader {
            gpu,
            positions,
            in_buf,
            out_buf,
            heap_buf,
            metadata_buf,
            staging_buf,
            bind_groups,
            bind_group_layouts,
            pipeline,
        }
    }
}

struct ObservationShader {}


pub struct GPURunner {
    lts: TransitionSystem,
    gpu: Rc<GPUCommon>,
    graph_bind_group: wgpu::BindGroup,

    clause_shader: ClauseShader,
    challenge_shader: ChallengeShader,
}

impl GPURunner {
    pub async fn with_lts(lts: TransitionSystem) -> Result<Self> {
        let gpu = Rc::new(GPUCommon::new().await?);
        let (graph_bind_group_layout, graph_bind_group) = lts.bind_group(&gpu.device);

        let clause_shader = ClauseShader::new(Rc::clone(&gpu));
        // let conjunction_shader = ...
        let challenge_shader = ChallengeShader::new(Rc::clone(&gpu), &graph_bind_group_layout);

        Ok(GPURunner {
            lts,
            gpu,
            graph_bind_group,

            clause_shader,
            challenge_shader,
        })
    }

    /*
    pub async fn execute_gpu(&self) -> Result<()> {
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Game build encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Game build clause compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.clause_shader.pipeline);
            cpass.set_bind_group(0, &self.clause_shader.bind_group, &[]);

            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.clause_shader.buf_b, 0, &self.clause_shader.staging_buf, 0, 280);

        self.gpu.queue.submit(Some(encoder.finish()));
        let (sender, receiver) = futures_intrusive::channel::shared::channel(1);
        let staging_slice = self.clause_shader.staging_buf.slice(..);
        staging_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.try_send(v).expect("Channel should be writable");
        });

        self.gpu.device.poll(wgpu::Maintain::Wait);
        receiver.receive().await.expect("Channel should not be closed")?;

        let raw_data = staging_slice.get_mapped_range();
        let data: &[Position] = bytemuck::cast_slice(&raw_data);

        println!("{:?}", data);

        drop(raw_data);
        self.clause_shader.staging_buf.unmap();

        Ok(())
    }
    */
    pub async fn execute_gpu(&self) -> Result<()> {
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Game build encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Game build challenge compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.challenge_shader.pipeline);
            cpass.set_bind_group(0, &self.challenge_shader.bind_groups[0], &[]);
            cpass.set_bind_group(1, &self.challenge_shader.bind_groups[1], &[]);
            cpass.set_bind_group(2, &self.graph_bind_group, &[]);

            cpass.dispatch_workgroups(1, 1, 1);
        }
        let staging_buf = &self.challenge_shader.staging_buf;
        encoder.copy_buffer_to_buffer(&self.challenge_shader.out_buf, 0, staging_buf, 0, self.challenge_shader.out_buf.size());
        encoder.copy_buffer_to_buffer(&self.challenge_shader.heap_buf, 0,
                                      staging_buf, self.challenge_shader.out_buf.size(),
                                      self.challenge_shader.heap_buf.size());
        encoder.copy_buffer_to_buffer(&self.challenge_shader.metadata_buf, 0,
                                      staging_buf, self.challenge_shader.out_buf.size() + self.challenge_shader.heap_buf.size(),
                                      self.challenge_shader.metadata_buf.size());

        self.gpu.queue.submit(Some(encoder.finish()));
        let (sender, receiver) = futures_intrusive::channel::shared::channel(1);
        let staging_slice = self.challenge_shader.staging_buf.slice(..);
        staging_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.try_send(v).expect("Channel should be writable");
        });

        self.gpu.device.poll(wgpu::Maintain::Wait);
        receiver.receive().await.expect("Channel should not be closed")?;

        let raw_data = staging_slice.get_mapped_range();
        let pos_out: &[ConjunctionPosition] = bytemuck::cast_slice(&raw_data[..self.challenge_shader.out_buf.size() as usize]);
        let heap: &[LinkedList] = bytemuck::cast_slice(
            &raw_data[self.challenge_shader.out_buf.size() as usize .. (self.challenge_shader.out_buf.size() + self.challenge_shader.heap_buf.size()) as usize]);
        let metadata: &Metadata = bytemuck::from_bytes(
            &raw_data[(self.challenge_shader.out_buf.size() + self.challenge_shader.heap_buf.size()) as usize ..]);

        println!("Positions: {:?}", pos_out);
        println!("Heap: {:?}", heap);
        println!("Metadata: {:?}", metadata);

        drop(raw_data);
        self.challenge_shader.staging_buf.unmap();

        Ok(())
    }
}
