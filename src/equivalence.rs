//! Types for handling results from multiple comparisons.
//!
//! This module contains types for conveniently handling the resulting energy budgets
//! when comparing all processes in an LTS with each other.
//!
//! It allows inspecting equivalence relations for the various types of equivalences.

use std::ops::Deref;

use disjoint_sets::UnionFind;
use rustc_hash::FxHashMap;

use crate::{EnergyArray, Energy};
use crate::gamebuild::AttackPosition;


/// Information on the comparisons the algorithm was started with.
#[derive(Debug, Clone)]
pub struct StartInfo {
    /// Total number of processes in the LTS.
    pub n_processes: usize,
    /// All requested comparisons as positions for two processes.
    pub starting_points: Vec<AttackPosition>,
}

impl StartInfo {
    pub fn new(starting_points: Vec<AttackPosition>, n_processes: usize) -> Self {
        StartInfo {
            n_processes,
            starting_points,
        }
    }

    /// Associate this StartInfo with the resulting energies.
    pub fn equivalence(self, energies: Vec<EnergyArray>) -> Equivalence {
        Equivalence::new(self, energies)
    }
}


/// Contains results of comparing multiple processes, allows inspecting equivalence relations.
#[derive(Debug, Clone)]
pub struct Equivalence {
    /// Info about what process pairs were compared.
    pub start_info: StartInfo,
    /// Computed energies for all starting positions.
    pub energies: Vec<EnergyArray>,
    /// For a starting position (an [`AttackPosition`] comparing two processes)
    /// maps to its index in `start_info.starting_points` as well as `energies`.
    pub pos_to_idx: FxHashMap<AttackPosition, usize>,
    /// Minimization used to reduce the LTS.
    pub minimization: Option<Vec<usize>>,
}

impl Equivalence {
    /// Create object by associating computed energies with starting information.
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
            minimization: None,
        }
    }

    /// Apply the minimization mapping, if any, to a process `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is outside the range of mapped processes.
    #[inline]
    fn mapped(&self, p: u32) -> u32 {
        match &self.minimization {
            Some(mapping) => mapping[p as usize] as u32,
            None => p,
        }
    }

    /// Retrieve energies associated with the position `(p, q)`.
    ///
    /// If the position `(p, q)` was not included in the initial starting points for game
    /// graph generation, returns `None`.
    ///
    /// # Panics
    ///
    /// Panics if `p` or `q` is outside the range of processes.
    pub fn energies(&self, p: u32, q: u32) -> Option<&EnergyArray> {
        self.energies_minimized(self.mapped(p), self.mapped(q))
    }

    /// Energies for comparing the processes `p` and `q` in the minimized LTS.
    ///
    /// # Panics
    ///
    /// Panics if `p` or `q` is outside the range of minimized processes.
    fn energies_minimized(&self, p: u32, q: u32) -> Option<&EnergyArray> {
        assert!((p as usize) < self.start_info.n_processes);
        assert!((q as usize) < self.start_info.n_processes);
        let pos = AttackPosition { p, q: vec![q] };
        self.pos_to_idx.get(&pos)
            .map(|idx| self.energies.get(*idx).expect("pos_to_idx should hold valid indices"))
    }

    /// Returns `true`, if process `p` is covered by process 'q' according to a given
    /// `equivalence`. That is, `p <= q` for the preorder induced by `equivalence`.
    ///
    /// This can be used to test equivalences that are not inherently symmetric.
    /// The function [`equiv`](Equivalence::equiv) instead requires the preorder
    /// to hold in both directions:
    ///
    /// `self.equiv(p, q, equ) == true` if and only if `self.preorder(p, q, equ) && self.preorder(q, p, equ)`
    ///
    /// # Panics
    ///
    /// Panics if `p` or `q` is outside the range of processes.
    pub fn preorder(&self, p: u32, q: u32, equivalence: &Energy) -> bool {
        let p = self.mapped(p);
        let q = self.mapped(q);
        if p == q {
            return true;
        }
        self.energies_minimized(p, q).is_some_and(|e| e.test_equivalence(equivalence))
    }

    /// Returns `true`, if processes `p` and `q` are equivalent according to the given equivalence.
    /// In contrast to [`preorder`](Equivalence::preorder),
    /// this function requires the equivalence in both directions,
    /// so the order of `p` and `q` doesn't matter.
    ///
    /// # Panics
    ///
    /// Panics if `p` or `q` is outside the range of processes.
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
        let mut union = UnionFind::new(self.start_info.n_processes);
        for (pos, energy) in self.start_info.starting_points.iter().zip(&self.energies) {
            let p = pos.p;
            let q = pos.q[0];
            let e2 = &self.energies_minimized(q, p)
                .expect("Symmetric position not included in starting points for energy game");
            if energy.test_equivalence(equivalence) && e2.test_equivalence(equivalence) {
                union.union(p as usize, q as usize);
            }
        }
        let relation = EquivalenceRelation::from(union);
        match &self.minimization {
            Some(mapping) => relation.with_mapping(mapping),
            None => relation,
        }
    }

    pub(crate) fn set_minimization(&mut self, minimization: Vec<usize>) {
        self.minimization = Some(minimization);
    }
}


/// Equivalence relation over processes.
#[derive(Debug, Clone)]
pub struct EquivalenceRelation {
    /// The underlying union-find data structure.
    pub union: UnionFind,
}

impl EquivalenceRelation {
    /// Returns all elements, grouped by their equivalence class.
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

    /// The number of equivalence classes.
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

    /// Returns an iterator over the equivalence class containing `proc`.
    pub fn class_of(&self, proc: usize) -> impl Iterator<Item=usize> + '_ {
        let class = self.find(proc);
        (0..self.len())
            .filter(move |i| self.find(*i) == class)
    }

    /// Create new EquivalenceRelation accounting for a mapping.
    ///
    /// This mapping could could be the bisimulation used for minimizing the LTS,
    /// which maps processes to their bisimulation classes.
    /// Applying a minimization returns an EquivalenceRelation
    /// refering to the original processes, not the minimized LTS.
    ///
    /// # Panics
    ///
    /// Panics, if any element of `mapping` is not included in the union,
    /// meaning it is less than `self.len()`.
    fn with_mapping(&self, mapping: &[usize]) -> Self {
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
