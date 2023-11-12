use std::fmt;

use serde::{Serialize, Deserialize};

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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


#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(into = "i8", from = "i8")]
pub enum Upd {
    #[default]
    Zero,
    Decrement,
    Min(u8),
}

impl fmt::Display for Upd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Upd::Zero => write!(f, "0"),
            Upd::Decrement => write!(f, "-1"),
            Upd::Min(idx) => write!(f, "Min({idx})"),
        }
    }
}

impl From<i8> for Upd {
    /// Update elements can be easily represented by integers:
    ///
    /// -1 => Decrement update
    /// 0  => No update
    /// 1  => Min update between current position and first position
    /// 2  => Min update between current position and second position
    /// etc.
    fn from(n: i8) -> Upd {
        match n {
            0 => Upd::Zero,
            -1 => Upd::Decrement,
            i8::MIN..=-2 => panic!("Invalid update entry"),
            i => Upd::Min(i as u8),
        }
    }
}

impl From<Upd> for i8 {
    fn from(upd: Upd) -> i8 {
        match upd {
            Upd::Zero => 0,
            Upd::Decrement => -1,
            Upd::Min(idx) => idx as i8,
        }
    }
}


#[repr(transparent)]
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(into = "Vec<Upd>", from = "Vec<Upd>")]
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
                Upd::Zero => 0,
                Upd::Decrement => 1,
                Upd::Min(idx) => {
                    assert!(*idx <= 14, "Min update indices must be at most 14");
                    *idx as u32 + 1
                },
            };
            data |= encoded << shift;
            shift += 4;
        }
        Update(data)
    }
}

impl From<Vec<Upd>> for Update {
    fn from(vec: Vec<Upd>) -> Self {
        vec.as_slice().into()
    }
}

#[macro_export]
macro_rules! update {
    ( $( $x:expr ),* ) => {
        Update::from([ $( $x.into(), )* ].as_slice())
    };
}

impl From<&Update> for Vec<Upd> {
    fn from(update: &Update) -> Vec<Upd> {
        let mut shifted = update.0.clone();
        let mut vec = Vec::with_capacity(8);
        loop {
            vec.push(match shifted & 0xf {
                0 => Upd::Zero,
                1 => Upd::Decrement,
                idx => Upd::Min(idx as u8 - 1),
            });
            shifted >>= 4;
            if shifted == 0 {
                return vec;
            }
        }
    }
}

impl From<Update> for Vec<Upd> {
    fn from(update: Update) -> Vec<Upd> {
        (&update).into()
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


#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn test_energy() {
        assert_eq!(Energy::default(), Energy::zero());
        assert_eq!(Energy::zero().0, 0);
        assert_eq!(Energy::zeroed(), Energy::zero(), "Zeroable trait from bytemuck");
        assert_eq!(Energy::from(23409).0, 23409);

        let parts = vec![0, 3, 1, 0, 2];
        let energy = Energy::from(parts.as_slice());
        assert_eq!(energy, energy![0, 3, 1, 0, 2]);
        assert_eq!(energy, energy![0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(energy.0, 0x21c, "Bitpacked value: each element is 2 bits wide");
    }

    #[test]
    fn test_upd() {
        assert_eq!(Upd::default(), Upd::Zero); 
        assert_eq!(Upd::from(0), Upd::Zero);
        assert_eq!(Upd::from(-1), Upd::Decrement);
        assert_eq!(Upd::from(1), Upd::Min(1));
        assert_eq!(Upd::from(8), Upd::Min(8));

        assert_eq!(i8::from(Upd::Zero), 0);
        assert_eq!(i8::from(Upd::Decrement), -1);
        assert_eq!(i8::from(Upd::Min(1)), 1);
        assert_eq!(i8::from(Upd::Min(8)), 8);
    }

    #[test]
    fn test_update() {
        assert_eq!(Update::default(), Update::zero());
        assert_eq!(Update::zero().0, 0);
        assert_eq!(Update::zeroed(), Update::zero(), "Zeroable trait from bytemuck");
        assert_eq!(Update::from(1528).0, 1528); 

        let parts = vec![Upd::Zero, Upd::Decrement, Upd::Min(1), Upd::Min(8)];
        let update = Update::from(parts.as_slice());
        assert_eq!(update, Update::from(parts.clone()));
        assert_eq!(update, update![0, -1, 1, 8]);
        assert_eq!(update, update![Upd::default(), -1, 1, Upd::Min(8)]);
        assert_eq!(update, update![0, -1, 1, 8, 0, 0, 0, 0], "Trailing zeros don't change the value");
        assert_eq!(update.0, 0x9210, "Bit representation: Min(8) -> 0x9, Min(1) -> 0x2, -1 -> 1, 0 -> 0");
        assert_eq!(Vec::<Upd>::from(update), parts, "Disassemble Update into Vec<Upd>");
    }

    #[test]
    fn update_serde() {
        let parts = vec![Upd::Zero, Upd::Decrement, Upd::Min(1), Upd::Min(8)];
        let update = Update::from(parts.as_slice());
        assert_eq!(update, serde_json::from_str("[0, -1, 1, 8]").unwrap(), "Deserialization from JSON");
        assert_eq!(serde_json::to_string(&update).unwrap(), "[0,-1,1,8]", "Serialization into JSON");

        assert_eq!(Update::zero(), serde_json::from_str("[]").unwrap(), "Deserialize empty list");
        assert_eq!(Update::zero(), serde_json::from_str("[0]").unwrap());
        assert_eq!(Update::zero(), serde_json::from_str("[0, 0]").unwrap());
        assert_eq!(serde_json::to_string(&Update::zero()).unwrap(), "[0]", "Serialization of empty update");
    }
}
