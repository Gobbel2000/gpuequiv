use std::iter;
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

struct ConjunctionShader {}

struct ChallengeShader {}

struct ObservationShader {}


pub struct GPURunner {
    lts: TransitionSystem,
    gpu: Rc<GPUCommon>,
    graph_bind_group: wgpu::BindGroup,

    clause_shader: ClauseShader,
}

impl GPURunner {
    pub async fn with_lts(lts: TransitionSystem) -> Result<Self> {
        let gpu = Rc::new(GPUCommon::new().await?);
        let (_graph_bind_group_layout, graph_bind_group) = lts.bind_group(&gpu.device);

        let clause_shader = ClauseShader::new(Rc::clone(&gpu));

        Ok(GPURunner {
            lts,
            gpu,
            graph_bind_group,

            clause_shader,
        })
    }

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
}
