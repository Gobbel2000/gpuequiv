use wgpu::{Buffer, Device, Queue};

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

        let available = adapter.limits();
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        // Get as much buffer storage space as we can
        // Otherwise use default limits with high compatibility
        let max_size = available.max_buffer_size.min(available.max_storage_buffer_binding_size.into());
        required_limits.max_storage_buffer_binding_size = max_size as u32;
        required_limits.max_buffer_size = required_limits.max_buffer_size.max(max_size);
        log::debug!("Setting max_storage_buffer_binding_size to {max_size}");

        Ok(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits,
                },
                None,
            )
            .await?)
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
