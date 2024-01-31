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
}

impl Equivalence {
    pub fn new(start_info: StartInfo, energies: Vec<EnergyArray>) -> Self {
        Equivalence { start_info, energies }
    }

    pub fn relation(&self, equivalence: &Energy) -> EquivalenceRelation {
        let mut union = self.start_info.starting_equivalence.clone().into_inner();
        for (pos, energy) in self.start_info.starting_points.iter().zip(&self.energies) {
            if energy.test_equivalence(equivalence) {
                union.union(pos.p as usize, pos.q[0] as usize);
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
