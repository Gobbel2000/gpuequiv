use std::fmt;
use std::cmp::Ordering;
use std::iter;

use rustc_hash::FxHashSet;
use ndarray::{Array2, ArrayView2, ArrayView1, Axis, aview1, ArrayBase, Ix2, Data};

macro_rules! fn_get_conf {
    () => {
        pub fn get_conf(&self) -> EnergyConf {
            self.conf
        }
    };
}

pub trait FromEnergyConf<T>: Sized {
    type Error: std::fmt::Debug;
    fn from_conf(a: T, conf: EnergyConf) -> Result<Self, Self::Error>;
}

pub trait IntoEnergyConf<T>: Sized {
    type Error;
    fn into_conf(self, conf: EnergyConf) -> Result<T, Self::Error>;
}

impl<T, U> IntoEnergyConf<U> for T 
where
    U: FromEnergyConf<T>,
{
    type Error = U::Error;
    fn into_conf(self, conf: EnergyConf) -> Result<U, Self::Error> {
        U::from_conf(self, conf)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnergyConf {
    pub elements: u32,
    pub max: u32,
}

impl EnergyConf {
    pub const STANDARD: Self = EnergyConf { elements: 6, max: 3 };
    pub const SILENT_STEP: Self = EnergyConf { elements: 8, max: 2 };
    pub const ONE_DIMENSION: Self = EnergyConf { elements: 1, max: u32::MAX };

    // Number of u32's required to store all elements
    pub fn energy_size(&self) -> u32 {
        //sizes or add padding at the end of each word.
        let bits = self.energy_bits() * self.elements;
        bits.div_ceil(u32::BITS)
    }

    // Number of bits required to store one element
    pub fn energy_bits(&self) -> u32 {
        (u32::BITS - self.max.leading_zeros())
            .next_power_of_two()
    }

    pub fn energy_mask(&self) -> u32 {
        u32::MAX >> (u32::BITS - self.energy_bits())
    }

    pub fn update_size(&self) -> u32 {
        let bits = self.update_bits() * self.elements;
        bits.div_ceil(u32::BITS)
    }

    pub fn update_bits(&self) -> u32 {
        (u32::BITS - (self.elements + 2).leading_zeros())
            .next_power_of_two()
    }

    pub fn update_mask(&self) -> u32 {
        u32::MAX >> (u32::BITS - self.update_bits())
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Energy {
    data: Vec<u32>,
    conf: EnergyConf,
}

impl Energy {
    pub fn new(values: &[u32], conf: EnergyConf) -> Option<Self> {
        Some(Self {
            data: pack_energy(values, conf)?,
            conf,
        })
    }

    pub fn from_raw_data(data: &[u32], conf: EnergyConf) -> Self {
        assert_eq!(data.len(), conf.energy_size() as usize);
        Self {
            data: data.to_vec(),
            conf,
        }
    }

    pub fn raw_data(&self) -> &[u32] {
        &self.data
    }

    pub fn zero(conf: EnergyConf) -> Self {
        Energy {
            data: vec![0; conf.energy_size() as usize],
            conf,
        }
    }

    pub fn to_vec(&self) -> Vec<u32> {
        self.into()
    }

    fn_get_conf!();
}

impl PartialOrd for Energy {
    // Element-wise partial comparison between two energy tuples
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        assert_eq!(self.conf.elements, other.conf.elements, "Incompatible Energy configurations");
        let mut less = true;
        let mut greater = true;
        for (e0, e1) in iter::zip(self.to_vec(), other.to_vec()) {
            match e0.cmp(&e1) {
                Ordering::Less => greater = false,
                Ordering::Greater => less = false,
                Ordering::Equal => {},
            }
        }
        match (less, greater) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }
}

fn pack_energy(values: &[u32], conf: EnergyConf) -> Option<Vec<u32>> {
    if values.len() > conf.elements as usize {
        return None;
    }
    let mut energy = vec![0; conf.energy_size() as usize];
    let mut shift = 0;
    let mut word = 0;
    for val in values {
        if *val > conf.max {
            return None;
        }
        energy[word] |= val << shift;    
        shift += conf.energy_bits();
        if shift + conf.energy_bits() > u32::BITS {
            shift = 0;
            word += 1;
        }
    }
    Some(energy)
}

impl From<&Energy> for Vec<u32> {
    fn from(energy: &Energy) -> Vec<u32> {
        let mut vec = Vec::with_capacity(energy.conf.elements as usize);
        let mut shift = 0;
        let mut word = 0;
        for _ in 0..energy.conf.elements as usize {
            vec.push((energy.data[word] >> shift) & energy.conf.energy_mask());
            shift += energy.conf.energy_bits();
            if shift >= u32::BITS {
                shift = 0;
                word += 1;
            }
        }
        vec
    }
}

impl fmt::Display for Energy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, " ")?;
        for val in self.to_vec() {
            if val == self.conf.max {
                write!(f, "∞ ")?;
            } else {
                write!(f, "{val} ")?;
            };
        }
        Ok(())
    }
}

// Energy tuples representing notions of equivalence.
// From: B. Bisping - Process Equivalence Problems as Energy Games
pub mod std_equivalences {
    use super::{Energy, EnergyConf};
    use std::sync::OnceLock;

    macro_rules! econst {
        ($name:ident => $($x:expr),*) => {
            pub fn $name() -> &'static Energy {
                static ONCE: OnceLock<Energy> = OnceLock::new();
                ONCE.get_or_init(|| {
                    Energy::new(&[$( $x, )*], EnergyConf::STANDARD).unwrap()
                })
            }
        }
    }
    econst!(enabledness => 1, 1);
    econst!(traces => 3, 1);
    econst!(failures => 3, 2, 0, 0, 1, 1);
    econst!(failure_traces => 3, 3, 3, 0, 1, 1);
    econst!(readiness => 3, 2, 1, 1, 1, 1);
    econst!(readiness_traces => 3, 3, 3, 1, 1, 1);
    econst!(revivals => 3, 2, 1, 0, 1, 1);
    econst!(impossible_futures => 3, 2, 0, 0, 3, 1);
    econst!(possible_futures => 3, 2, 3, 3, 3, 1);
    econst!(simulation => 3, 3, 3, 3, 0, 0);
    econst!(ready_simulation => 3, 3, 3, 3, 1, 1);
    econst!(nested_2simulation => 3, 3, 3, 3, 3, 1);
    econst!(bisimulation => 3, 3, 3, 3, 3, 3);
}

#[derive(Debug, Clone, Eq)]
pub struct EnergyArray {
    array: Array2<u32>,
    conf: EnergyConf,
}

impl EnergyArray {
    pub fn empty(conf: EnergyConf) -> Self {
        Self::zero(0, conf)
    }

    pub fn zero(n: usize, conf: EnergyConf) -> Self {
        let array = Array2::zeros((n, conf.energy_size() as usize));
        Self {
            array,
            conf,
        }
    }

    pub fn get(&self, idx: usize) -> Option<Energy> {
        if idx >= self.array.nrows() {
            return None;
        }
        let row = self.array.row(idx);
        let data = row.as_slice().expect("Array should be contiguous and in standard layout");
        Some(Energy::from_raw_data(data, self.conf))
    }

    pub fn set(&mut self, idx: usize, energy: Energy) -> Option<()> {
        if energy.conf != self.conf {
            return None;
        }
        if idx >= self.array.nrows() {
            return None;
        }
        self.array.row_mut(idx).assign(&aview1(&energy.data));
        Some(())
    }

    pub fn data(&self) -> &[u8] {
        let slice_u32 = self.array.as_slice().expect("Array should be contiguous and in standard layout");
        bytemuck::cast_slice(slice_u32)
    }

    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    pub fn n_energies(&self) -> usize {
        self.array.nrows()
    }

    pub fn into_inner(self) -> Array2<u32> {
        self.array
    }

    pub fn view(&self) -> ArrayView2<u32> {
        self.array.view()
    }

    pub fn iter(&self) -> impl Iterator<Item=Energy> + '_ {
        self.array.rows().into_iter().map(|row|
            Energy::from_raw_data(
                row.as_slice().expect("Array should be contiguous and in standard layout"),
                self.conf)
            )
    }

    /// Test whether these winning budgets confirm the given equivalence or refute it.
    ///
    /// Returns `true` if the equivalence holds, otherwise `false`.
    ///
    /// This assumes that `self` is a set of winning budgets
    /// calculated as the final result from solving the energy game.
    /// If associated to an attacker game position `(p, {q})`,
    /// this function gives insight into the equivalences between processes `p` and `q`.
    ///
    /// # Panics
    ///
    /// If `self` and `equivalence` do not have the same [`EnergyConf`].
    pub fn test_equivalence(&self, equivalence: &Energy) -> bool {
        assert_eq!(self.conf, equivalence.conf, "Incompatible Energy configurations");
        // The equivalence is disproven if the equivalence's tuple lies within the set of winning
        // energies, meaning there exists a formula within the subset of formulas characterizing
        // the equivalence, that distinguishes the processes.
        // Because this array holds the minimal points of an upward-closed set, the equivalence
        // tuple must lie above (ore equal) one of these points to be in that set.

        // If this array is empty, this method always returns `true`, meaning all equivalences
        // hold, because in the energy game the attacker could not reach a distinguishing position.
        !self.iter().any(|e| &e <= equivalence)
    }

    /// Constructs an EnergyArray struct from array data.
    /// Care must be taken to only provide valid data. This function does not check whether the
    /// data in the array adheres to conf.max and whether all padding bits are zeroed.
    ///
    /// The only check is that the number of columns in the array matches the width required for
    /// the configuration. If it doesn't match, this function panics.
    pub fn from_array(array: Array2<u32>, conf: EnergyConf) -> Self {
        assert_eq!(array.ncols(), conf.energy_size() as usize);
        Self { array, conf }
    }

    fn_get_conf!();
}

impl<S> PartialEq<ArrayBase<S, Ix2>> for EnergyArray
where
    S: Data<Elem=u32>
{
    fn eq(&self, other: &ArrayBase<S, Ix2>) -> bool {
        if self.array.shape() != other.shape() {
            return false;
        }
        let set: FxHashSet<ArrayView1<u32>> = self.array.rows().into_iter().collect();
        for row in other.rows() {
            if !set.contains(&row) {
                return false;
            }
        }
        true
    }
}

impl<S> PartialEq<EnergyArray> for ArrayBase<S, Ix2>
where
    S: Data<Elem=u32>
{
    fn eq(&self, other: &EnergyArray) -> bool {
        other.eq(self)
    }
}

impl PartialEq for EnergyArray {
    // Compare two EnergyArrays without regard to ordering
    fn eq(&self, other: &Self) -> bool {
        if self.conf != other.conf {
            return false;
        }
        self.eq(&other.array)
    }
}

impl fmt::Display for EnergyArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = "[".to_string();
        for energy in self.iter() {
            out.push_str(&format!("{energy}\n "));
        }
        if out.len() >= 2 {
            out.replace_range(out.len() - 2.., "]");
        } else {
            out.push(']');
        }
        write!(f, "{out}")
    }
}

impl From<EnergyArray> for Vec<Energy> {
    fn from(array: EnergyArray) -> Vec<Energy> {
        array.iter().collect()
    }
}

impl From<&EnergyArray> for Vec<Vec<u32>> {
    fn from(array: &EnergyArray) -> Vec<Vec<u32>> {
        array.iter().map(|energy| (&energy).into())
            .collect()
    }
}

impl TryFrom<&[Energy]> for EnergyArray {
    type Error = &'static str;
    fn try_from(energies: &[Energy]) -> Result<Self, Self::Error> {
        let conf = energies.first().ok_or("Empty Energy list")?.get_conf();
        if !energies.iter().skip(1).all(|u| conf == u.get_conf()) {
            return Err("Mismatched energy configurations");
        }
        let flat: Vec<u32> = energies.iter()
            .flat_map(|u| &u.data)
            .copied()
            .collect();
        let array = Array2::from_shape_vec((energies.len(), conf.energy_size() as usize), flat)
            .map_err(|_| "Error while creating array")?;
        Ok(Self { array, conf })
    }
}

impl FromEnergyConf<&[Energy]> for EnergyArray {
    type Error = &'static str;
    fn from_conf(energies: &[Energy], conf: EnergyConf) -> Result<Self, Self::Error> {
        if let Some(energy) = energies.first() {
            if energy.get_conf() != conf {
                return Err("EnergyConf in update list doesn't match provided EnergyConf");
            }
        } else {
            // Return empty array
            return Ok(Self::zero(0, conf));
        }
        Self::try_from(energies)
    }
}

impl FromEnergyConf<&[Vec<u32>]> for EnergyArray {
    type Error = &'static str;
    fn from_conf(energies: &[Vec<u32>], conf: EnergyConf) -> Result<Self, Self::Error> {
        let mut array = Array2::zeros((energies.len(), conf.energy_size() as usize));
        for (mut row, energy_vals) in array.rows_mut().into_iter().zip(energies) {
            let packed = pack_energy(energy_vals, conf).ok_or("Updates dont' match configuration")?;
            row.assign(&aview1(&packed));
        }
        Ok(Self { array, conf })
    }
}


/// A single element of an [`Update`]-tuple
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Upd {
    #[default]
    Zero,
    Decrement,
    Min(u32),
}

impl Upd {
    fn packed_repr(&self) -> u32 {
        match self {
            Upd::Zero => 0,
            Upd::Decrement => 1,
            Upd::Min(idx) => idx + 1,
        }
    }
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


impl From<i32> for Upd {
    /// Update elements can be easily represented by integers:
    ///
    /// -1 => Decrement update
    /// 0  => No update
    /// 1  => Min update between current position and first position
    /// 2  => Min update between current position and second position
    /// etc.
    fn from(n: i32) -> Upd {
        match n {
            0 => Upd::Zero,
            -1 => Upd::Decrement,
            i32::MIN..=-2 => panic!("Invalid update entry"),
            i => Upd::Min(i as u32),
        }
    }
}

impl From<Upd> for i32 {
    fn from(upd: Upd) -> i32 {
        match upd {
            Upd::Zero => 0,
            Upd::Decrement => -1,
            Upd::Min(idx) => idx as i32,
        }
    }
}


#[derive(Clone, PartialEq)]
pub struct Update{
    data: Vec<u32>,
    conf: EnergyConf,
}

impl Update {
    pub fn new(values: &[Upd], conf: EnergyConf) -> Option<Self> {
        Some(Self {
            data: pack_upd(values, conf)?,
            conf,
        })
    }

    pub fn from_raw_data(data: &[u32], conf: EnergyConf) -> Self {
        assert_eq!(data.len(), conf.update_size() as usize);
        Self {
            data: data.to_vec(),
            conf,
        }
    }

    pub fn raw_data(&self) -> &[u32] {
        &self.data
    }

    pub fn zero(conf: EnergyConf) -> Self {
        Self {
            data: vec![0; conf.update_size() as usize],
            conf,
        }
    }

    pub fn to_vec(&self) -> Vec<Upd> {
        self.into()
    }

    fn_get_conf!();
}

fn pack_upd(values: &[Upd], conf: EnergyConf) -> Option<Vec<u32>> {
    if values.len() > conf.elements as usize {
        return None;
    }
    let mut update = vec![0; conf.update_size() as usize];
    let mut shift = 0;
    let mut word = 0;
    for val in values {
        if let Upd::Min(idx) = val {
            // Upd::Min is 1-indexed
            if *idx > conf.elements {
                return None;
            }
        }
        update[word] |= val.packed_repr() << shift;    
        shift += conf.update_bits();
        if shift + conf.update_bits() > u32::BITS {
            shift = 0;
            word += 1;
        }
    }
    Some(update)
}

impl From<&Update> for Vec<Upd> {
    fn from(update: &Update) -> Vec<Upd> {
        let mut vec = Vec::with_capacity(update.conf.elements as usize);
        let mut shift = 0;
        let mut word = 0;
        for _ in 0..update.conf.elements as usize {
            let val = (update.data[word] >> shift) & update.conf.update_mask();
            vec.push(match val {
                0 => Upd::Zero,
                1 => Upd::Decrement,
                idx => Upd::Min(idx - 1),
            });

            shift += update.conf.update_bits();
            if shift >= u32::BITS {
                shift = 0;
                word += 1;
            }
        }
        vec
    }
}

impl From<Update> for Vec<Upd> {
    fn from(update: Update) -> Vec<Upd> {
        (&update).into()
    }
}

impl From<&Update> for Vec<i32> {
    fn from(update: &Update) -> Vec<i32> {
        let upd_vec: Vec<Upd> = update.into();
        upd_vec.iter().map(|&upd| upd.into())
            .collect()
    }
}

impl fmt::Debug for Update {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vec = self.to_vec();
        write!(f, "<{}", vec.first().copied().unwrap_or_default())?;
        for e in self.to_vec().iter().skip(1) {
            write!(f, " {e}")?;
        }
        write!(f, ">")
    }
}

#[macro_export]
macro_rules! update {
    ( $( $x:expr ),* ) => {
        vec![ $( Upd::from($x), )* ]
    };
}


#[derive(Debug, Clone, PartialEq)]
pub struct UpdateArray {
    pub(crate) array: Array2<u32>,
    conf: EnergyConf,
}

impl UpdateArray {
    pub fn empty(conf: EnergyConf) -> Self {
        Self::zero(0, conf)
    }

    pub fn zero(n: usize, conf: EnergyConf) -> Self {
        let array = Array2::zeros((n, conf.update_size() as usize));
        Self {
            array,
            conf,
        }
    }

    pub fn repeat(update: Update, n: usize) -> Self {
        let row = aview1(&update.data);
        let view = row.broadcast((n, update.data.len())).unwrap();
        Self {
            array: view.to_owned(),
            conf: update.conf,
        }
    }

    pub fn push_n(&mut self, update: Update, n: usize) {
        assert_eq!(update.conf, self.conf, "Incompatible energy configurations");
        let row = aview1(&update.data);
        let view = row.broadcast((n, update.data.len())).unwrap();
        self.array.append(Axis(0), view).unwrap();
    }

    pub fn get(&self, idx: usize) -> Option<Update> {
        if idx >= self.array.nrows() {
            return None;
        }
        let row = self.array.row(idx);
        let data = row.as_slice().expect("Array should be contiguous and in standard layout");
        Some(Update::from_raw_data(data, self.conf))
    }

    pub fn set(&mut self, idx: usize, update: Update) -> Option<()> {
        if update.conf != self.conf {
            return None;
        }
        if idx >= self.array.nrows() {
            return None;
        }
        self.array.row_mut(idx).assign(&aview1(&update.data));
        Some(())
    }

    pub fn data(&self) -> &[u8] {
        let slice_u32 = self.array.as_slice().expect("Array should be contiguous and in standard layout");
        bytemuck::cast_slice(slice_u32)
    }

    pub fn into_inner(self) -> Array2<u32> {
        self.array
    }

    pub fn n_updates(&self) -> usize {
        self.array.nrows()
    }

    pub fn iter(&self) -> impl Iterator<Item=Update> + '_ {
        self.array.rows().into_iter().map(|row|
            Update::from_raw_data(row.as_slice()
            .expect("Array should be contiguous and in standard layout"), self.conf))
    }

    /// Constructs an [`UpdateArray`] struct from array data.
    /// Care must be taken to only provide valid data. This function does not check whether the
    /// data in the array adheres to `conf.max` and whether all padding bits are zeroed.
    ///
    /// # Panics
    ///
    /// The only check is that the number of columns in the array matches the width required for
    /// the configuration. If it doesn't match, this function panics.
    pub fn from_array(array: Array2<u32>, conf: EnergyConf) -> Self {
        assert_eq!(array.ncols(), conf.update_size() as usize);
        Self { array, conf }
    }

    fn_get_conf!();
}

impl From<UpdateArray> for Vec<Update> {
    fn from(array: UpdateArray) -> Vec<Update> {
        array.iter().collect()
    }
}

impl From<&UpdateArray> for Vec<Vec<i32>> {
    fn from(array: &UpdateArray) -> Vec<Vec<i32>> {
        array.iter().map(|update| (&update).into())
            .collect()
    }
}

impl TryFrom<&[Update]> for UpdateArray {
    type Error = &'static str;
    fn try_from(updates: &[Update]) -> Result<Self, Self::Error> {
        let conf = updates.first().ok_or("Empty Update list")?.get_conf();
        if !updates.iter().skip(1).all(|u| conf == u.get_conf()) {
            return Err("Mismatched energy configurations");
        }
        let flat: Vec<u32> = updates.iter()
            .flat_map(|u| &u.data)
            .copied()
            .collect();
        let array = Array2::from_shape_vec((updates.len(), conf.update_size() as usize), flat)
            .map_err(|_| "Error while creating array")?;
        Ok(Self { array, conf })
    }
}

impl FromEnergyConf<&[Update]> for UpdateArray {
    type Error = &'static str;
    fn from_conf(updates: &[Update], conf: EnergyConf) -> Result<Self, Self::Error> {
        if let Some(update) = updates.first() {
            if update.get_conf() != conf {
                return Err("EnergyConf in update list doesn't match provided EnergyConf");
            }
        } else {
            // Return empty array
            return Ok(Self::zero(0, conf));
        }
        Self::try_from(updates)
    }
}

impl FromEnergyConf<&[Vec<Upd>]> for UpdateArray {
    type Error = &'static str;
    fn from_conf(updates: &[Vec<Upd>], conf: EnergyConf) -> Result<Self, Self::Error> {
        let mut array = Array2::zeros((updates.len(), conf.update_size() as usize));
        for (mut row, upds) in array.rows_mut().into_iter().zip(updates) {
            let packed = pack_upd(upds, conf).ok_or("Updates don't match configuration")?;
            row.assign(&aview1(&packed));
        }
        Ok(Self { array, conf })
    }
}

impl FromEnergyConf<&[&[Upd]]> for UpdateArray {
    type Error = &'static str;
    fn from_conf(updates: &[&[Upd]], conf: EnergyConf) -> Result<Self, Self::Error> {
        let mut array = Array2::zeros((updates.len(), conf.update_size() as usize));
        for (mut row, upds) in array.rows_mut().into_iter().zip(updates) {
            let packed = pack_upd(upds, conf).ok_or("Updates dont' match configuration")?;
            row.assign(&aview1(&packed));
        }
        Ok(Self { array, conf })
    }
}

impl FromEnergyConf<&[Vec<i32>]> for UpdateArray {
    type Error = &'static str;
    fn from_conf(updates: &[Vec<i32>], conf: EnergyConf) -> Result<Self, Self::Error> {
        let upds: Vec<Vec<Upd>> = updates.iter().map(|row| row.iter()
                .map(|num| Upd::from(*num))
                .collect())
            .collect();
        Self::from_conf(upds.as_slice(), conf)
    }
}

impl FromEnergyConf<&[&[i32]]> for UpdateArray {
    type Error = &'static str;
    fn from_conf(updates: &[&[i32]], conf: EnergyConf) -> Result<Self, Self::Error> {
        let upds: Vec<Vec<Upd>> = updates.iter().map(|row| row.iter()
                .map(|num| Upd::from(*num))
                .collect())
            .collect();
        Self::from_conf(upds.as_slice(), conf)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_conf() {
        let c = EnergyConf { elements: 4, max: 3 };
        assert_eq!(c.energy_size(), 1);
        assert_eq!(c.energy_bits(), 2);
        assert_eq!(c.energy_mask(), 0x3);
        assert_eq!(c.update_size(), 1);
        assert_eq!(c.update_bits(), 4);
        assert_eq!(c.update_mask(), 0xf);

        let c = EnergyConf { elements: 6, max: 100 };
        assert_eq!(c.energy_size(), 2);
        assert_eq!(c.energy_bits(), 8);
        assert_eq!(c.energy_mask(), 0xff);
        assert_eq!(c.update_size(), 1);
        assert_eq!(c.update_bits(), 4);
        assert_eq!(c.update_mask(), 0xf);

        let c = EnergyConf { elements: 30, max: 3 };
        assert_eq!(c.energy_size(), 2);
        assert_eq!(c.energy_bits(), 2);
        assert_eq!(c.energy_mask(), 0x3);
        assert_eq!(c.update_size(), 8);
        assert_eq!(c.update_bits(), 8);
        assert_eq!(c.update_mask(), 0xff);

        let c = EnergyConf { elements: 4, max: u32::MAX };
        assert_eq!(c.energy_size(), 4);
        assert_eq!(c.energy_bits(), 32);
        assert_eq!(c.energy_mask(), u32::MAX);
        assert_eq!(c.update_size(), 1);
        assert_eq!(c.update_bits(), 4);
        assert_eq!(c.update_mask(), 0xf);
    }

    #[test]
    fn test_energy() {
        let conf = EnergyConf { elements: 12, max: 15 };
        let values = vec![0, 1, 2, 3, 4, 5, 6, 7,
                          8, 13, 14, 15];
        let energy = Energy::new(values.as_slice(), conf).unwrap();
        assert_eq!(energy.data, vec![0x76543210, 0xfed8]);
        assert_eq!(energy.to_vec(), values);

        let zero = Energy::zero(conf);
        assert_eq!(zero.data, vec![0, 0]);
        assert_eq!(zero.to_vec(), vec![0; 12]);
    }

    #[test]
    fn test_energy_array() {
        let conf = EnergyConf { elements: 6, max: 3 };
        let mut array = EnergyArray::zero(4, conf);
        let row = array.get(2).unwrap();
        assert_eq!(row.data, vec![0]);
        let energy = Energy::new(&vec![0, 1, 2, 0, 3], conf).unwrap();
        array.set(2, energy.clone());
        assert_eq!(array.data(), &[0, 0, 0, 0, 0, 0, 0, 0, 0x24, 0x3, 0, 0, 0, 0, 0, 0]); 
        assert_eq!(energy.data, vec![0x324]);
        assert_eq!(energy.to_vec(), vec![0, 1, 2, 0, 3, 0]);
        let a2 = EnergyArray::zero(0, conf);
        assert_eq!(a2.array.nrows(), 0);
    }

    #[test]
    fn test_energy_ord() {
        let c = EnergyConf { elements: 6, max: 3 };
        let e0 = Energy::zero(c);
        let e1 = Energy::new(&vec![1, 1, 1, 1, 1, 1], c).unwrap();
        let e2 = Energy::new(&vec![0, 0, 0, 0, 0, 1], c).unwrap();
        let e3 = Energy::new(&vec![1, 0, 0, 0, 0, 0], c).unwrap();
        assert!(e0 < e1);
        assert!(e0 < e2);
        assert!(e2 < e1);
        assert_eq!(e2.partial_cmp(&e3), None); // e2 and e3 are incomparable
        assert_eq!(e3.partial_cmp(&e2), None);
        assert_eq!(e2.partial_cmp(&e2), Some(Ordering::Equal));
    }

    #[test]
    fn test_equivalences() {
        // Test these to make sure the macro works properly
        assert_eq!(std_equivalences::enabledness(),
                   &Energy::new(&vec![1, 1], EnergyConf::STANDARD).unwrap());
        assert_eq!(std_equivalences::bisimulation(),
                   &Energy::new(&vec![3, 3, 3, 3, 3, 3], EnergyConf::STANDARD).unwrap());
    }

    #[test]
    fn test_upd() {
        assert_eq!(Upd::default(), Upd::Zero); 
        assert_eq!(Upd::from(0), Upd::Zero);
        assert_eq!(Upd::from(-1), Upd::Decrement);
        assert_eq!(Upd::from(1), Upd::Min(1));
        assert_eq!(Upd::from(8), Upd::Min(8));

        assert_eq!(i32::from(Upd::Zero), 0);
        assert_eq!(i32::from(Upd::Decrement), -1);
        assert_eq!(i32::from(Upd::Min(1)), 1);
        assert_eq!(i32::from(Upd::Min(8)), 8);
    }

    #[test]
    fn test_update() {
        let conf = EnergyConf { elements: 12, max: 3 };
        let values = vec![Upd::Zero, Upd::Decrement, Upd::Min(1), Upd::Min(2),
                          Upd::Min(3), Upd::Min(4), Upd::Min(5), Upd::Min(6),
                          Upd::Min(9), Upd::Min(10), Upd::Min(11), Upd::Min(12)];
        let update = Update::new(values.as_slice(), conf).unwrap();
        assert_eq!(update.data, vec![0x76543210, 0xdcba]);
        assert_eq!(update.to_vec(), values);
        let nums: Vec<i32> = (&update).into();
        assert_eq!(nums, vec![0, -1, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12]);
        assert_eq!(update.to_vec(), update![Upd::Zero, -1, 1, 2,
                                            Upd::Min(3), Upd::Min(4), Upd::Min(5), Upd::Min(6),
                                            9, 10, 11, Upd::Min(12)]);

        let zero = Update::zero(conf);
        assert_eq!(zero.data, vec![0, 0]);
        assert_eq!(zero.to_vec(), vec![Upd::Zero; 12]);
    }

    #[test]
    fn test_update_array() {
        let conf = EnergyConf { elements: 6, max: 3 };
        let array = UpdateArray::from_conf(vec![vec![Upd::Zero, Upd::Decrement, Upd::Min(2)]].as_slice(), conf).unwrap();
        assert_eq!(array, UpdateArray::from_conf(vec![vec![0, -1, 2]].as_slice(), conf).unwrap());
    }
}
