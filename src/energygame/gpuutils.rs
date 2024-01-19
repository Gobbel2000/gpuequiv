use wgpu::{Buffer, Device, Queue};
use wgpu::util::DeviceExt;

use crate::error::{Result, Error};

// Common handles and data for managing the GPU device
#[derive(Debug)]
pub(crate) struct GPUCommon {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
}

impl GPUCommon {
    pub(crate) async fn new() -> Result<GPUCommon> {
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
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await?)
    }
}

pub(crate) trait GPUGraph {
    type Weight: bytemuck::NoUninit;
    fn csr(&self) -> (Vec<u32>, Vec<u32>, Vec<Self::Weight>);

    fn buffers(&self, device: &Device) -> (Buffer, Buffer, Buffer) {
        let (c, r, w) = self.csr();
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

    fn bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Input graph bind group layout"),
            entries: &[
                bgl_entry(0, true), // graph column indices
                bgl_entry(1, true), // graph row offsets
                bgl_entry(2, true), // graph edge weights
            ],
        })
    }

    fn bind_group_from_parts(
        buffers: &(Buffer, Buffer, Buffer),
        layout: &wgpu::BindGroupLayout,
        device: &Device,
    ) -> wgpu::BindGroup {
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

    fn bind_group(&self, device: &Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        let layout = Self::bind_group_layout(device);
        let buffers = self.buffers(device);
        let bind_group = Self::bind_group_from_parts(&buffers, &layout, device);
        (layout, bind_group)
    }
}


// Convenience function for creating bind group layout entries
#[inline]
pub(crate) fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
pub(crate) fn buffer_fits<T>(vec: &Vec<T>, buf: &Buffer) -> bool {
    vec.len() * std::mem::size_of::<T>() <= buf.size() as usize
}
