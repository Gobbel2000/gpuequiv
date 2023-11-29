use std::{fmt, error, result};

#[derive(Debug)]
pub enum Error {
    NoAdapter,
    NoDevice(wgpu::RequestDeviceError),
    Overflow,
    BufferMap(wgpu::BufferAsyncError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAdapter => write!(f, "could not get GPU adapter"),
            Error::NoDevice(source) => write!(f, "could not get GPU device: {}", source),
            Error::Overflow => write!(f, "overflow occured during calculation, input size too large"),
            Error::BufferMap(source) => source.fmt(f),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::NoDevice(source) => Some(source),
            Error::BufferMap(source) => Some(source),
            _ => None,
        }
    }
}

impl From<wgpu::RequestDeviceError> for Error {
    fn from(err: wgpu::RequestDeviceError) -> Error {
        Error::NoDevice(err)
    }
}

impl From<wgpu::BufferAsyncError> for Error {
    fn from(err: wgpu::BufferAsyncError) -> Error {
        Error::BufferMap(err)
    }
}

pub type Result<T> = result::Result<T, Error>;

