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
    /// containing all available transitions from a process `start`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is outside the range of processes of this LTS.
    #[inline]
    pub fn adj(&self, start: u32) -> &[Transition] {
        &self.adj[start as usize]
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

    pub async fn compare_multiple(&self, processes: &[u32]) -> Result<Equivalence> {
        let (reduced, minimization) = self.bisimilar_minimize();
        let mut equivalence = reduced.compare_multiple_unminimized(processes).await?;
        equivalence.set_minimization(minimization);
        Ok(equivalence)
    }

    pub async fn compare_multiple_unminimized(&self, processes: &[u32]) -> Result<Equivalence> {
        let (builder, start_info) = GameBuild::compare_multiple(self, processes);
        let mut game = EnergyGame::standard_reach(builder.game);
        game.run().await?;
        Ok(start_info.equivalence(game.energies))
    }

    /// Find all equivalences for all process pairs in the LTS.
    ///
    /// Returns an [`Equivalence`] struct which can be used to explore the results.
    /// See the documentation of [`Equivalence`] for further details.
    /// The second return element is the mapping used for minimization.
    pub async fn equivalences(&self) -> Result<Equivalence> {
        let (reduced, minimization) = self.bisimilar_minimize();
        let mut equivalence = reduced.equivalences_unminimized().await?;
        equivalence.set_minimization(minimization);
        Ok(equivalence)
    }

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
