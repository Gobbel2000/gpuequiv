pub mod energygame;
pub mod gamebuild;
mod energy;
mod error;
mod bisimulation;
mod equivalence;

// Re-exports
pub use energy::*;
pub use error::*;
pub use equivalence::*;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::result;

use gamebuild::GameBuild;
use energygame::EnergyGame;

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

    #[inline]
    pub fn adj(&self, start: u32) -> &[Transition] {
        &self.adj[start as usize]
    }

    #[inline]
    pub fn inner(&self) -> &[Vec<Transition>] {
        &self.adj
    }

    #[inline]
    pub fn into_inner(self) -> Vec<Vec<Transition>> {
        self.adj
    }

    pub async fn compare(&self, p: u32, q: u32) -> Result<EnergyArray> {
        let builder = GameBuild::compare(self, p, q);
        let mut game = EnergyGame::standard_reach(builder.game);
        game.run().await?;
        Ok(game.energies.into_iter().next().unwrap())
    }

    pub async fn compare_multiple(&self, processes: &[u32]) -> Result<(Equivalence, Vec<usize>)> {
        let (reduced, minimization) = self.bisimilar_minimize();
        let equivalence = reduced.compare_multiple_unminimized(processes).await?;
        Ok((equivalence, minimization))
    }

    pub async fn compare_multiple_unminimized(&self, processes: &[u32]) -> Result<Equivalence> {
        let (builder, start_info) = GameBuild::compare_multiple(self, processes, true);
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

    /// Create LTS from adjacency list. This function assumes that the lists in `adj` are already
    /// sorted by label. If not, the algorithm will produce incorrect results. Use
    /// [`TransitionSystem::from()`] instead to ensure correct sorting.
    pub fn from_adj_unchecked(adj: Vec<Vec<Transition>>) -> Self {
        TransitionSystem { adj }
    }

    fn sort_labels(&mut self) {
        for row in self.adj.iter_mut() {
            row.sort_by_key(|t| t.label);
        }
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
