pub mod energy;
pub mod energygame;
pub mod gamebuild;
pub mod error;
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
use energygame::{EnergyGame, GameGraph};

use rustc_hash::FxHashMap;


#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone)]
pub struct TransitionSystem {
    pub adj: Vec<Vec<Transition>>,
}

type Edge = (u32, u32, i32);

impl TransitionSystem {
    pub fn new(n_vertices: u32, edges: Vec<Edge>) -> Self {
        let mut adj = vec![vec![]; n_vertices as usize];
        for (from, to, label) in edges {
            adj[from as usize].push(Transition { process: to, label });
        }
        TransitionSystem::from(adj)
    }

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

    pub fn sort_labels(&mut self) {
        for row in self.adj.iter_mut() {
            row.sort_by_key(|t| t.label);
        }
    }

    pub fn n_vertices(&self) -> u32 {
        self.adj.len() as u32
    }

    pub fn build_game_graph(&self, p: u32, q: u32) -> GameGraph {
        let builder = GameBuild::compare(self, p, q);
        builder.game
    }

    pub async fn winning_budgets(&self, p: u32, q: u32) -> Result<Vec<EnergyArray>> {
        let game_graph = self.build_game_graph(p, q);
        let mut game = EnergyGame::standard_reach(game_graph);
        game.run().await?;
        Ok(game.energies)
    }

    pub async fn equivalences(&self) -> Result<(Equivalence, Vec<usize>)> {
        let (reduced, mapping) = self.bisimilar_minimize();
        Ok((reduced.equivalences_unminimized().await?, mapping))
    }

    pub async fn equivalences_unminimized(&self) -> Result<Equivalence> {
        let (builder, start_info) = GameBuild::compare_all(self, true);
        let mut game = EnergyGame::standard_reach(builder.game);
        game.run().await?;
        Ok(start_info.equivalence(game.energies))
    }
}

impl From<Vec<Vec<Transition>>> for TransitionSystem {
    // Create new LTS from adjacency list, ensuring that it is properly sorted
    fn from(adj: Vec<Vec<Transition>>) -> Self {
        let mut lts = TransitionSystem { adj };
        lts.sort_labels();
        lts
    }
}

impl FromIterator<Edge> for TransitionSystem {
    fn from_iter<I: IntoIterator<Item=Edge>>(iter: I) -> Self {
        let mut adj = vec![];
        for (from, to, label) in iter {
            let from = from as usize;
            if from >= adj.len() {
                adj.resize(from + 1, vec![]);
            }
            adj[from].push(Transition { process: to, label });
        }
        TransitionSystem::from(adj)
    }
}
