//! A GPU-accelerated implementation of an algorithm to find all equivalences
//! of processes in the Linear-time Branching-time Spectrum.
//!
//! This project is part of my bachelor's thesis at Technische Universität Berlin.
//! The [accompanying thesis text](https://github.com/Gobbel2000/thesis-gpuequiv)
//! can be consulted as a more in-depth documentation.
//!
//! This is a Rust crate implementing the
//! [Spectroscopy algorithm](https://arxiv.org/abs/2303.08904) by B. Bisping
//! with a focus on performance and scalability.
//! To this end, the most critical parts of the algorithm are accelerated by GPU
//! compute shaders.
//! The [wgpu](https://github.com/gfx-rs/wgpu) crate is used
//! for interfacing with the system's GPU API.
//! Shaders are written in the WebGPU Shading Language
//! ([WGSL](https://gpuweb.github.io/gpuweb/wgsl/)).
//! These technologies are based on the up-and-coming WebGPU API,
//! aiming to enable advanced access to GPUs in web browsers.
//! Therefore, this crate can be used in WebAssembly,
//! although it requires the browser to support the WebGPU API.
//!
//! Requires **Rust version** >= 1.73.
//!
//! ### Equivalences
//!
//! For an input Labeled Transition System, the algorithm will decide for any
//! process pair which of the behavioral equivalences in the following spectrum
//! hold and which don't. See [`std_equivalences`].
//!
//! * Enabledness
//! * Trace Equivalence
//! * Failures
//! * Failure Traces
//! * Readiness
//! * Readiness Traces
//! * Revivals
//! * Impossible Futures
//! * Possible Futures
//! * Simulation
//! * Ready Simulation
//! * Nested 2-Simulation
//! * Bisimulation
//!
//! # Usage
//!
//! Most of this crate's functionality can be accessed through the
//! [`TransitionSystem`] struct.
//! After constructing a transition system,
//! equivalences can be computed either between two processes
//! or between all process pairs of the system.
//!
//! ### Comparing two processes
//!
//! Using [`TransitionSystem::compare()`], the energies for comparing two
//! processes can be computed.
//! These energies encode the equivalences between them,
//! which can be tested using [`EnergyArray::test_equivalence()`]
//! with the definitions of the various equivalence notions from
//! [`std_equivalences`].
//! The following example calculates the equivalences between processes 0 and 4:
//!
//!
//! ```
//! use gpuequiv::{TransitionSystem, std_equivalences};
//!
//! # fn main() {
//! #     pollster::block_on(run());
//! # }
//! # async fn run() {
//! let (a, b, c) = (0, 1, 2);
//! let lts = TransitionSystem::from(vec![
//!     (0, 1, a), (1, 2, b), (1, 3, c),
//!     (4, 5, a), (5, 7, b), (4, 6, a), (6, 8, c),
//! ]);
//!
//! let energies = lts.compare(0, 4).await.unwrap();
//! // Process 4 does not simulates process 0
//! assert!(!energies.test_equivalence(std_equivalences::simulation()));
//! // Process 4 has all traces that process 0 has
//! assert!(energies.test_equivalence(std_equivalences::traces()));
//! # }
//! ```
//!
//! ### Comparing all processes in a system
//!
//! Handling the information for equivalences between all process pairs
//! is a bit more involved.
//! The function [`TransitionSystem::equivalences()`]
//! returns an [`Equivalence`] struct,
//! which can be used to explore equivalences between any two processes,
//! and even to construct equivalence relations.
//! Some of the ways in which that can be used are shown below:
//!
//! ```
//! use gpuequiv::{TransitionSystem, std_equivalences};
//!
//! # fn main() {
//! #     pollster::block_on(run());
//! # }
//! # async fn run() {
//! let (a, b, c) = (0, 1, 2);
//! let lts = TransitionSystem::from(vec![
//!     (0, 1, a), (1, 2, b), (1, 3, c),
//!     (4, 5, a), (5, 7, b), (4, 6, a), (6, 8, c),
//! ]);
//!
//! let equivalence = lts.equivalences().await.unwrap();
//!
//! // 4 is simulated by 0, but not the other way around, so they are not simulation-equivalent
//! assert!(equivalence.preorder(4, 0, std_equivalences::simulation()));
//! assert!(!equivalence.equiv(4, 0, std_equivalences::simulation()));
//!
//! // All process pairs can be queried
//! assert!(!equivalence.equiv(1, 5, std_equivalences::enabledness()));
//! assert!(equivalence.equiv(3, 7, std_equivalences::bisimulation()));
//!
//! // Inspect the trace equivalence relation
//! let traces = equivalence.relation(std_equivalences::traces());
//! assert_eq!(traces.count_classes(), 5);
//!
//! // All processes that are trace-equivalent with process 0
//! let mut class_of_0 = traces.class_of(0);
//! assert_eq!(class_of_0.next(), Some(0));
//! assert_eq!(class_of_0.next(), Some(4));
//! assert_eq!(class_of_0.next(), None);
//! # }
//! ```
//!
//! # Further Examples
//!
//! The [`examples`](https://github.com/Gobbel2000/gpuequiv/tree/master/examples)
//! directory in the repository contains further example files
//! on how this crate can be used.
//! [`compare.rs`](https://github.com/Gobbel2000/gpuequiv/blob/master/examples/compare.rs)
//! and
//! [`compare_all.rs`](https://github.com/Gobbel2000/gpuequiv/blob/master/examples/compare_all.rs)
//! further exemplify how this crate can be used to compare two or all processes,
//! [`energygame.rs`](https://github.com/Gobbel2000/gpuequiv/blob/master/examples/energygame.rs)
//! solves just an energy game, that was not created from an LTS.
//!
//! # Serde
//!
//! When compiled with the feature flag `serde` (disabled by default),
//! the structs [`TransitionSystem`]
//! and [`GameGraph`](crate::energygame::GameGraph)
//! implement serde's `Serialize` and `Deserialize` traits.

pub mod energygame;
pub mod gamebuild;
pub mod equivalence;
mod energy;
mod error;

// Re-exports
pub use energy::*;
pub use error::*;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::result;

use gamebuild::GameBuild;
use energygame::EnergyGame;
use equivalence::Equivalence;

use rustc_hash::FxHashMap;

/// A transition to the next process with a label. Part of a labeled transition system.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transition {
    /// Target process of this transition
    pub process: u32,
    /// Transition label encoded as i32
    ///
    /// If there are actual names (Strings), they should be stored in a separate list.
    pub label: i32,
}

/// Labeled Transition System (LTS) graph
///
/// The graph is represented by adjacency lists.
/// Each adjacent node (or process) is stored together with the transition label
/// in a [`Transition`] struct.
///
/// # Creation
///
/// A transition system can be constructed from 2-dimensional adjacency lists of transitions:
/// `Vec<Vec<Transition>>`.
/// A more convenient specification is listing all edges as 3-tuples `(u32, u32, i32)`,
/// which includes start and end nodes as well as the edge label.
/// The list does not need to be sorted.
/// For example:
///
/// ```
/// use gpuequiv::TransitionSystem;
///
/// // Define label names
/// let (a, b, c) = (0, 1, 2);
/// let lts = TransitionSystem::from(vec![
///     (0, 1, a),
///     (1, 2, b),
///     (1, 3, c),
///
///     (4, 5, a),
///     (5, 7, b),
///     (4, 6, a),
///     (6, 8, c),
/// ]);
/// ```
///
/// An LTS can also be read from a CSV file.
/// See [`from_csv_file()`](TransitionSystem::from_csv_file) for details.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TransitionSystem {
    adj: Vec<Vec<Transition>>,
}

type Edge = (u32, u32, i32);

impl TransitionSystem {
    /// The number of vertices (or processes) in this transition system.
    #[inline]
    pub fn n_vertices(&self) -> u32 {
        self.adj.len() as u32
    }

    /// Return the adjacency list,
    /// containing all available transitions from a process `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is outside the range of processes of this LTS.
    #[inline]
    pub fn adj(&self, p: u32) -> &[Transition] {
        &self.adj[p as usize]
    }

    /// Run the algorithm to compare processes `p` and `q`.
    ///
    /// Returns a set of energies that can be used to determine the equivalences that
    /// preorder `p` by `q`.
    ///
    /// # Example
    ///
    /// ```
    /// use gpuequiv::{TransitionSystem, std_equivalences};
    /// # fn main() {
    /// #     pollster::block_on(run());
    /// # }
    /// # async fn run() {
    /// let (a, b, c) = (0, 1, 2);
    /// let lts = TransitionSystem::from(vec![
    ///     (0, 1, a), (1, 2, b), (1, 3, c),
    ///     (4, 5, a), (5, 7, b), (4, 6, a), (6, 8, c),
    /// ]);
    ///
    /// let energies = lts.compare(0, 4).await.unwrap();
    /// // Process 4 does not simulates process 0
    /// assert!(!energies.test_equivalence(std_equivalences::simulation()));
    /// // Process 4 has all traces that process 0 has
    /// assert!(energies.test_equivalence(std_equivalences::traces()));
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// If no connection to a GPU could be made, an error is returned.
    ///
    /// # Panics
    ///
    /// Panics if `p` or `q` are outside the range of processes of this LTS.
    pub async fn compare(&self, p: u32, q: u32) -> Result<EnergyArray> {
        let (reduced, minimization) = self.bisimilar_minimize();
        let pm = minimization[p as usize] as u32;
        let qm = minimization[q as usize] as u32;
        if pm == qm { // p and q are bisimilar, we don't need to do anything
            return Ok(EnergyArray::empty(GameBuild::ENERGY_CONF));
        }
        reduced.compare_unminimized(pm, qm).await
    }

    /// The same as [`compare()`](TransitionSystem::compare),
    /// but without minimizing the LTS first.
    ///
    /// # Panics
    ///
    /// Panics if `p` or `q` are outside the range of processes of this LTS.
    pub async fn compare_unminimized(&self, p: u32, q: u32) -> Result<EnergyArray> {
        let builder = GameBuild::compare(self, p, q);
        let mut game = EnergyGame::standard_reach(builder.game);
        let energies = game.run().await?;
        Ok(energies.iter().next().unwrap().clone())
    }

    /// Find all equivalences for all process pairs in the LTS.
    ///
    /// Returns an [`Equivalence`] struct which can be used to explore the results.
    /// See the documentation of [`Equivalence`] for further details.
    ///
    /// # Example
    ///
    /// ```
    /// use gpuequiv::{TransitionSystem, std_equivalences};
    /// # fn main() {
    /// #     pollster::block_on(run());
    /// # }
    /// # async fn run() {
    /// let (a, b, c) = (0, 1, 2);
    /// let lts = TransitionSystem::from(vec![
    ///     (0, 1, a), (1, 2, b), (1, 3, c),
    ///     (4, 5, a), (5, 7, b), (4, 6, a), (6, 8, c),
    /// ]);
    ///
    /// let equivalence = lts.equivalences().await.unwrap();
    /// // All processes can now be compared
    /// assert!(!equivalence.preorder(0, 4, std_equivalences::simulation()));
    /// assert!(equivalence.equiv(0, 4, std_equivalences::traces()));
    /// assert!(!equivalence.equiv(1, 5, std_equivalences::enabledness()));
    /// assert!(equivalence.equiv(3, 7, std_equivalences::bisimulation()));
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// If no connection to a GPU could be made, an error is returned.
    pub async fn equivalences(&self) -> Result<Equivalence> {
        let (reduced, minimization) = self.bisimilar_minimize();
        let mut equivalence = reduced.equivalences_unminimized().await?;
        equivalence.set_minimization(minimization);
        Ok(equivalence)
    }

    /// The same as [`equivalences()`](TransitionSystem::equivalences),
    /// but without minimizing the LTS first.
    pub async fn equivalences_unminimized(&self) -> Result<Equivalence> {
        let (builder, start_info) = GameBuild::compare_all(self);
        let mut game = EnergyGame::standard_reach(builder.game);
        game.run().await?;
        Ok(start_info.equivalence(game.energies))
    }

    /// Build a labeled transition system from a CSV file.
    ///
    /// Each line should represent an edge of the graph,
    /// containing the three elements start process, end process and label.
    /// For example the line
    ///
    /// ```text
    /// 4,10,"enter"
    /// ````
    ///
    /// encodes a transition from process `4` to process `10` using the action `enter`.
    /// Quotes around the label name are optional.
    /// Note that these label names are not stored. They are replaced by unique integers.
    ///
    /// # Errors
    ///
    /// A [`CSVError`] is returned if a line does not contain 3 fields,
    /// if any of the first two fields can not be parsed as an unsigned integer,
    /// or if there were problems reading the file.
    pub fn from_csv_file<P: AsRef<Path>>(path: P) -> result::Result<Self, CSVError> {
        let file = File::open(path)?;
        let lines = io::BufReader::new(file).lines();
        Self::from_csv_lines(lines)
    }

    /// Build a transition system from an iterator of CSV lines.
    /// The format is described at [`from_csv_file()`](TransitionSystem::from_csv_file).
    pub fn from_csv_lines(lines: impl Iterator<Item=io::Result<String>>
    ) -> result::Result<Self, CSVError> {
        let mut adj = vec![];
        let mut labels: FxHashMap<String, i32> = FxHashMap::default();
        labels.insert("i".to_string(), 0);
        let mut max_label = 0;
        for l in lines {
            let l = l?;
            // Disassemble line
            let mut parts = l.splitn(3, ',');
            let from: usize = parts.next().ok_or(CSVError::MissingField)?.parse()?;
            let to: u32 = parts.next().ok_or(CSVError::MissingField)?.parse()?;
            let mut label = parts.next().ok_or(CSVError::MissingField)?;

            // Strip quotation marks
            if label.starts_with('"') && label.ends_with('"') && label.len() >= 2 {
                label = &label[1 .. label.len() - 1];
            }

            // Find the integer for the label
            let label_n = labels.get(label).copied().unwrap_or_else(|| {
                max_label += 1;
                labels.insert(label.to_string(), max_label);
                max_label
            });

            // If necessary, grow adjacency table to accommodate all mentioned nodes
            let max_node = from.max(to as usize);
            if max_node >= adj.len() {
                adj.resize(max_node + 1, vec![]);
            }
            adj[from].push(Transition { process: to, label: label_n });
        }
        Ok(TransitionSystem::from(adj))
    }

    /// Create a new, minimized LTS where bisimilar processes are consolidated.
    ///
    /// Returns a tuple containing the minimized system and the bisimulation used.
    /// This bisimulation is a mapping from original processes to the process in the minimized LTS
    /// that covers its bisimilarity class.
    pub fn bisimilar_minimize(&self) -> (TransitionSystem, Vec<usize>) {
        let (bisim, count) = self.signature_refinement();
        let mut adj = vec![vec![]; count];
        let mut represented = vec![false; count];
        for (process, &partition) in bisim.iter().enumerate() {
            if !represented[partition] {
                if adj.len() <= partition {
                    adj.resize(partition + 1, Vec::new());
                }
                adj[partition] = self.adj[process].iter()
                    .map(|transition| Transition {
                        process: bisim[transition.process as usize] as u32,
                        label: transition.label,
                    })
                    .collect();
                represented[partition] = true;
            }
        }
        (adj.into(), bisim)
    }

    /// Find strong bisimulation using signature refinement.
    /// The first return value is the partition, the second the number of bisimilarity classes.
    ///
    /// The partition list contains an index of the bisimulation equivalence class for each process.
    /// That is, if `p = signature_refinement()`, then `p[i] == p[j]` iff `i ~ j`.
    /// In other words, if and only if two processes get assigned the same partition index,
    /// they are bisimilar.
    ///
    /// This sequential algorithm is described by S. Blom and S. Orzan in
    /// "A Distributed Algorithm for Strong Bisimulation Reduction of State Spaces", 2002.
    pub fn signature_refinement(&self) -> (Vec<usize>, usize) {
        let mut partition = vec![0; self.n_vertices() as usize];
        let mut prev_count: usize = 0;
        let mut new_count: usize = 1;
        while prev_count != new_count {
            prev_count = new_count;
            new_count = 0;
            let signatures = self.signatures(&partition);
            let mut sigmap = FxHashMap::default();
            for sig in &signatures {
                sigmap.entry(sig).or_insert_with(|| {
                    new_count += 1;
                    new_count - 1
                });
            }
            for (part, sig) in partition.iter_mut().zip(&signatures) {
                *part = sigmap[&sig];
            }
        }
        (partition, new_count)
    }

    // Returns a set-valued signature for each process i:
    // sig[i] = {(a, ID) | i -a-> j and partition[j] == ID}
    fn signatures(&self, partition: &[usize]) -> Vec<Vec<(i32, usize)>> {
        self.adj.iter().map(|adj| {
                let mut sig: Vec<_> = adj.iter()
                    .map(|transition| (transition.label, partition[transition.process as usize]))
                    .collect();
                sig.sort_unstable();
                sig.dedup();
                sig
            })
            .collect()
    }

    /// Create LTS from adjacency lists.
    ///
    /// This function assumes that the lists in `adj` are already
    /// sorted by label. If not, the algorithm will produce incorrect results. Use
    /// [`TransitionSystem::from()`] instead to ensure correct sorting.
    pub fn from_adj_unchecked(adj: Vec<Vec<Transition>>) -> Self {
        TransitionSystem { adj }
    }

    fn sort_labels(&mut self) {
        for row in &mut self.adj {
            row.sort_by_key(|t| t.label);
        }
    }

    /// Return a reference to the inner adjacency lists.
    #[inline]
    pub fn inner(&self) -> &[Vec<Transition>] {
        &self.adj
    }

    /// Take ownership of the inner adjacency lists.
    #[inline]
    pub fn into_inner(self) -> Vec<Vec<Transition>> {
        self.adj
    }
}

impl From<Vec<Vec<Transition>>> for TransitionSystem {
    /// Create new LTS from adjacency lists, ensuring that they are properly sorted.
    fn from(adj: Vec<Vec<Transition>>) -> Self {
        let mut lts = TransitionSystem::from_adj_unchecked(adj);
        lts.sort_labels();
        lts
    }
}

impl From<Vec<Edge>> for TransitionSystem {
    fn from(edges: Vec<Edge>) -> Self {
        edges.into_iter().collect()
    }
}

impl FromIterator<Edge> for TransitionSystem {
    fn from_iter<I: IntoIterator<Item=Edge>>(iter: I) -> Self {
        let mut adj = vec![];
        for (from, to, label) in iter {
            let from = from as usize;
            let max_node = from.max(to as usize);
            if max_node >= adj.len() {
                adj.resize(max_node + 1, vec![]);
            }
            adj[from].push(Transition { process: to, label });
        }
        TransitionSystem::from(adj)
    }
}
