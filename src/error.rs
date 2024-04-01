use std::{fmt, io, num, error, result};

#[derive(Debug)]
pub enum Error {
    NoAdapter,
    NoDevice(wgpu::RequestDeviceError),
    BufferMap(wgpu::BufferAsyncError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAdapter => write!(f, "could not get GPU adapter"),
            Error::NoDevice(source) => write!(f, "could not get GPU device: {source}"),
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

/// Type alias for `Result<T, gpuequiv::Error>`
pub type Result<T> = result::Result<T, Error>;


#[derive(Debug)]
pub enum CSVError {
    IOError(io::Error),
    MissingField,
    ParseError(num::ParseIntError),
}

impl fmt::Display for CSVError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CSVError::IOError(source) => source.fmt(f),
            CSVError::ParseError(source) => source.fmt(f),
            CSVError::MissingField => write!(f, "insufficient number of values in line"),
        }
    }
}

impl error::Error for CSVError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            CSVError::IOError(source) => Some(source),
            CSVError::ParseError(source) => Some(source),
            _ => None,
        }
    }
}

impl From<io::Error> for CSVError {
    fn from(e: io::Error) -> Self {
        CSVError::IOError(e)
    }
}

impl From<num::ParseIntError> for CSVError {
    fn from(e: num::ParseIntError) -> Self {
        CSVError::ParseError(e)
    }
}

impl From<CSVError> for io::Error {
    fn from(e: CSVError) -> io::Error {
        match e {
            CSVError::IOError(source) => source,
            CSVError::ParseError(_) | CSVError::MissingField => {
                io::Error::new(io::ErrorKind::InvalidData, e)
            }
        }
    }
}
