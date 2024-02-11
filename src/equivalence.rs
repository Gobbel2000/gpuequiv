//! This module contains types for conveniently handling the resulting energy budgets
//! when comparing all processe in an LTS with each other.
//!
//! It allows inspecting equivalence relations for the various types of equivalences.

use std::ops::Deref;

use disjoint_sets::UnionFind;
use rustc_hash::FxHashMap;

use crate::{EnergyArray, Energy};
use crate::gamebuild::AttackPosition;

#[derive(Debug, Clone)]
pub struct StartInfo {
    pub starting_equivalence: EquivalenceRelation,
    pub starting_points: Vec<AttackPosition>,
}

impl StartInfo {
    pub fn new(starting_points: Vec<AttackPosition>, n_nodes: usize) -> Self {
        StartInfo {
            starting_equivalence: UnionFind::new(n_nodes).into(),
            starting_points,
        }
    }

    pub fn from_partition(starting_points: Vec<AttackPosition>, partition: &[usize]) -> Self {
        let mut union = UnionFind::new(partition.len());
        let mut representatives = FxHashMap::default();
        for (proc, part) in partition.iter().enumerate() {
            if let Some(equal_to) = representatives.get(part) {
                union.union(*equal_to, proc);
            } else {
                representatives.insert(part, proc);
            } 
        }
        StartInfo {
            starting_equivalence: union.into(),
            starting_points,
        }
    }

    pub fn equivalence(self, energies: Vec<EnergyArray>) -> Equivalence {
        Equivalence::new(self, energies)
    }
}


#[derive(Debug, Clone)]
pub struct Equivalence {
    pub start_info: StartInfo,
    pub energies: Vec<EnergyArray>,
    pos_to_idx: FxHashMap<AttackPosition, usize>,
}

impl Equivalence {
    pub fn new(start_info: StartInfo, mut energies: Vec<EnergyArray>) -> Self {
        assert!(energies.len() >= start_info.starting_points.len(), "Not enough energies");
        // Retain only the required energies
        energies.truncate(start_info.starting_points.len());
        energies.shrink_to_fit();

        let pos_to_idx: FxHashMap<AttackPosition, usize> = start_info.starting_points.iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i))
            .collect();
        Equivalence {
            start_info,
            energies,
            pos_to_idx,
        }
    }

    /// Retrieve energies associated with the position `(p, q)`.
    ///
    /// # Panics
    ///
    /// Panics if the position `(p, q)` was not included in the initial starting points for game
    /// graph generation.
    pub fn energies(&self, p: u32, q: u32) -> &EnergyArray {
        let pos = AttackPosition { p, q: vec![q] };
        let idx = self.pos_to_idx.get(&pos)
            .expect("Position not included in starting points for energy game");
        self.energies.get(*idx).expect("Energy list out of range")
    }

    /// Returns `true`, if process `p` is covered by process 'q' according to a given
    /// `equivalence`. That is, `p <= q` for the preorder induced by `equivalence`.
    ///
    /// This can be used to test equivalences that are not inherently symmetric. The function
    /// [`equiv`](Equivalence::equiv) instead requires the preorder to hold in both directions:
    ///
    /// `self.equiv(p, q, equ) == true` if and only if `self.preorder(p, q, equ) && self.preorder(q, p, equ)`
    ///
    /// # Panics
    ///
    /// Panics if the position `(p, q)` was not included in the initial starting points for game
    /// graph generation.
    pub fn preorder(&self, p: u32, q: u32, equivalence: &Energy) -> bool {
        if p == q {
            return true;
        }
        self.energies(p, q).test_equivalence(equivalence)
    }

    /// Returns `true`, if processes `p` and `q` are equivalent according to the given equivalence.
    /// In contrast to [`preorder`](Equivalence::preorder), this function requires the equivalence in both directions, so
    /// the order of `p` and `q` doesn't matter.
    ///
    /// # Panics
    ///
    /// Panics if the position `(p, q)` was not included in the initial starting points for game
    /// graph generation.
    pub fn equiv(&self, p: u32, q: u32, equivalence: &Energy) -> bool {
        self.preorder(p, q, equivalence) && self.preorder(q, p, equivalence)
    }

    /// Create the full equivalence relation for a given equivalence.
    ///
    /// Any processes `p` and `q` will be in the same class if `self.equiv(p, q, equ) == true`.
    ///
    /// # Panics
    ///
    /// Panics if the initial starting points for game graph generation didn't include the full
    /// symmetric closure.
    pub fn relation(&self, equivalence: &Energy) -> EquivalenceRelation {
        let mut union = self.start_info.starting_equivalence.clone().into_inner();
        for (pos, energy) in self.start_info.starting_points.iter().zip(&self.energies) {
            let p = pos.p;
            let q = pos.q[0];
            let e2 = &self.energies(q, p);
            if energy.test_equivalence(equivalence) && e2.test_equivalence(equivalence) {
                union.union(p as usize, q as usize);
            }
        }
        union.into()
    }
}


#[derive(Debug, Clone)]
pub struct EquivalenceRelation {
    pub union: UnionFind,
}

impl EquivalenceRelation {
    pub fn with_mapping(&self, mapping: &[usize]) -> Self {
        let mut union = UnionFind::new(mapping.len());
        let mut representative = FxHashMap::default();
        for (proc, minimized) in mapping.iter().enumerate() {
            let part = self.find(*minimized);
            if let Some(class) = representative.get(&part) {
                union.union(proc, *class);
            } else {
                representative.insert(part, proc);
            }
        }
        union.into()
    }

    pub fn get_classes(&self) -> Vec<Vec<usize>> {
        let mut class_idx = FxHashMap::default();
        let mut classes = Vec::new();
        for i in 0..self.len() {
            let class = self.find(i);
            let idx = class_idx.entry(class)
                .or_insert_with(|| {
                    classes.push(Vec::new());
                    classes.len() - 1
                });
            classes[*idx].push(i);
        }
        classes
    }

    pub fn count_classes(&self) -> u32 {
        let mut class_counted = vec![false; self.len()];
        let mut count = 0;
        for i in 0..self.len() {
            let class = self.find(i);
            if !class_counted[class] {
                count += 1;
                class_counted[class] = true;
            }
        }
        count
    }

    pub fn class_of(&self, proc: usize) -> impl Iterator<Item=usize> + '_ {
        let class = self.find(proc);
        (0..self.len())
            .filter(move |i| self.find(*i) == class)
    }

    pub fn into_inner(self) -> UnionFind {
        self.union
    }
}

// Allows UnionFind methods to be called on EquivalenceRelation structs
impl Deref for EquivalenceRelation {
    type Target = UnionFind;
    fn deref(&self) -> &Self::Target {
        &self.union
    }
}

impl From<UnionFind> for EquivalenceRelation {
    fn from(union: UnionFind) -> Self {
        EquivalenceRelation { union }
    }
}
