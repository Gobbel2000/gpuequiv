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

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transition {
    pub process: u32,

    // Transition labels encoded as i32:
    // 0 => τ
    // k => Channel index k, k ∈ ℕ
    // -k => Co-Action of k, k ∈ ℕ
    //
    // Actual names (Strings) should be stored in a separate list
    pub label: i32,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TransitionSystem {
    adj: Vec<Vec<Transition>>,
}

type Edge = (u32, u32, i32);

impl TransitionSystem {
    pub fn from_csv_lines(lines: impl Iterator<Item=io::Result<String>>) -> result::Result<Self, CSVError> {
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

    pub fn from_csv_file<P: AsRef<Path>>(path: P) -> result::Result<Self, CSVError> {
        let file = File::open(path)?;
        let lines = io::BufReader::new(file).lines();
        Self::from_csv_lines(lines)
    }

    #[inline]
    pub fn n_vertices(&self) -> u32 {
        self.adj.len() as u32
    }

    /// # Panics
    ///
    /// Panics if `p` is outside the range of processes of this LTS.
    #[inline]
    pub fn adj(&self, start: u32) -> &[Transition] {
        &self.adj[start as usize]
    }

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

    /// # Panics
    ///
    /// Panics if `p` or `q` are outside the range of processes of this LTS.
    pub async fn compare_unminimized(&self, p: u32, q: u32) -> Result<EnergyArray> {
        let builder = GameBuild::compare(self, p, q);
        let mut game = EnergyGame::standard_reach(builder.game);
        let energies = game.run().await?;
        Ok(energies.iter().next().unwrap().clone())
    }

    pub async fn compare_multiple(&self, processes: &[u32]) -> Result<(Equivalence, Vec<usize>)> {
        let (reduced, minimization) = self.bisimilar_minimize();
        let equivalence = reduced.compare_multiple_unminimized(processes).await?;
        Ok((equivalence, minimization))
    }

    pub async fn compare_multiple_unminimized(&self, processes: &[u32]) -> Result<Equivalence> {
        let (builder, start_info) = GameBuild::compare_multiple(self, processes);
        let mut game = EnergyGame::standard_reach(builder.game);
        game.run().await?;
        Ok(start_info.equivalence(game.energies))
    }

    pub async fn equivalences(&self) -> Result<(Equivalence, Vec<usize>)> {
        let (reduced, minimization) = self.bisimilar_minimize();
        let equivalence = reduced.equivalences_unminimized().await?;
        Ok((equivalence, minimization))
    }

    pub async fn equivalences_unminimized(&self) -> Result<Equivalence> {
        let (builder, start_info) = GameBuild::compare_all(self);
        let mut game = EnergyGame::standard_reach(builder.game);
        game.run().await?;
        Ok(start_info.equivalence(game.energies))
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

    /// Create LTS from adjacency list. This function assumes that the lists in `adj` are already
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

    #[inline]
    pub fn inner(&self) -> &[Vec<Transition>] {
        &self.adj
    }

    #[inline]
    pub fn into_inner(self) -> Vec<Vec<Transition>> {
        self.adj
    }
}

impl From<Vec<Vec<Transition>>> for TransitionSystem {
    // Create new LTS from adjacency list, ensuring that it is properly sorted
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
