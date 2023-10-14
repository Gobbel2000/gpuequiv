use std::fmt;

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Energy(pub u32);

impl From<u32> for Energy {
    fn from(data: u32) -> Self {
        Energy(data)
    }
}

impl From<&[u32]> for Energy {
    fn from(array: &[u32]) -> Self {
        assert!(array.len() <= 16, "Energies must have at most 16 elements");
        let mut data = 0;
        let mut shift = 0;
        for e in array {
            assert!(*e <= 3, "Energy elements must not be higher than 3");
            data |= e << shift;
            shift += 2;
        }
        Energy(data)
    }
}

// Allow directly constructing energies with various size, similar to the vec! macro:
// energy![0, 1, 2, 3] is equivalent to Energy::from([0, 1, 2, 3].as_slice()).
// Any number of energies up to 16 is allowed. energy![] constructs an all-zero energy.
#[macro_export]
macro_rules! energy {
    ( $( $x:expr ),* ) => {
        Energy::from([ $( $x, )* ].as_slice())
    };
}

impl Energy {
    pub fn zero() -> Self {
        Energy(0)
    }
}

impl Default for Energy {
    fn default() -> Self {
        Energy::zero()
    }
}

impl fmt::Debug for Energy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut shifted = self.0.clone();
        loop {
            match shifted & 0x3 {
                3 => write!(f, "∞")?,
                n => write!(f, "{n}")?,
            };
            shifted >>= 2;
            if shifted == 0 {
                return Ok(());
            }
        }
    }
}

// Enable bytemucking for filling buffers
unsafe impl bytemuck::Zeroable for Energy {}
unsafe impl bytemuck::Pod for Energy {}


#[derive(Clone, Copy, Default)]
pub enum Upd {
    #[default]
    None,
    Decrement,
    Min(u8),
}

impl fmt::Display for Upd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Upd::None => write!(f, "0"),
            Upd::Decrement => write!(f, "-1"),
            Upd::Min(idx) => write!(f, "Min({idx})"),
        }
    }
}


#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Update(pub u32);

impl Update {
    pub fn zero() -> Self {
        Update(0)
    }

    pub fn to_vec(&self) -> Vec<Upd> {
        self.into()
    }
}

impl Default for Update {
    fn default() -> Self {
        Update::zero()
    }
}

impl From<u32> for Update {
    fn from(data: u32) -> Self {
        Update(data)
    }
}

impl From<&[Upd]> for Update {
    fn from(array: &[Upd]) -> Self {
        assert!(array.len() <= 8, "Updates must have at most 8 elements");
        let mut data = 0;
        let mut shift = 0;
        for e in array {
            let encoded = match e {
                Upd::None => 0,
                Upd::Decrement => 1,
                Upd::Min(idx) => {
                    assert!(*idx <= 13, "Min update indices must be at most 13");
                    *idx as u32 + 2
                },
            };
            data |= encoded << shift;
            shift += 4;
        }
        Update(data)
    }
}

#[macro_export]
macro_rules! update {
    ( $( $x:expr ),* ) => {
        Update::from([ $( $x, )* ].as_slice())
    };
}


impl From<&Update> for Vec<Upd> {
    fn from(update: &Update) -> Vec<Upd> {
        let mut shifted = update.0.clone();
        let mut vec = Vec::with_capacity(8);
        loop {
            vec.push(match shifted & 0xf {
                0 => Upd::None,
                1 => Upd::Decrement,
                idx => Upd::Min(idx as u8),
            });
            shifted >>= 4;
            if shifted == 0 {
                return vec;
            }
        }
    }
}

impl fmt::Debug for Update {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vec = self.to_vec();
        write!(f, "<{}", vec.first().copied().unwrap_or_default())?;
        for e in self.to_vec().iter().skip(1) {
            write!(f, " {}", e)?;
        }
        write!(f, ">")
    }
}

// Enable bytemucking for filling buffers
unsafe impl bytemuck::Zeroable for Update {}
unsafe impl bytemuck::Pod for Update {}
