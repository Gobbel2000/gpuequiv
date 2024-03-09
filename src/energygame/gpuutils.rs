use std::sync::OnceLock;

use wgpu::{Buffer, Device, Queue};

use crate::error::{Result, Error};

// Common handles and data for managing the GPU device
#[derive(Debug)]
pub(crate) struct GPUCommon {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
}

impl GPUCommon {
    pub(crate) async fn get_gpu() -> Result<&'static GPUCommon> {
        // Save GPU handle in static memory so it only needs to be accessed once.
        // GPUCommon.get_device() takes 90ms to run,
        // so it is best to not call this more often than necessary.
        static ONCE: OnceLock<GPUCommon> = OnceLock::new();
        if let Some(gpu) = ONCE.get() {
            return Ok(gpu);
        }

        // GPU has not yet been requested.
        // Because get_device() needs to be async, we cannot put the initialization code into
        // `ONCE.get_or_init()`. This means if two threads were to call this function at the same
        // time, the device may be requested more than once, which is not really problematic.
        let (device, queue) = Self::get_device().await?;

        Ok(ONCE.get_or_init(|| GPUCommon { device, queue }))
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
pub(crate) fn buffer_fits<T>(slice: &[T], buf: &Buffer) -> bool {
    std::mem::size_of_val(slice) as u64 <= buf.size()
}
