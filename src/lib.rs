pub mod energy;
pub mod energygame;
pub mod gamebuild;
pub mod error;

// Re-exports
pub use energy::*;
pub use error::*;

use gamebuild::GameBuild;
use energygame::{EnergyGame, GameGraph};


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

pub struct TransitionSystem {
    pub adj: Vec<Vec<Transition>>,
}

impl TransitionSystem {
    pub fn new(n_vertices: u32, edges: Vec<(u32, u32, i32)>) -> Self {
        let mut adj = vec![vec![]; n_vertices as usize];
        for (from, to, label) in edges {
            adj[from as usize].push(Transition { process: to, label });
        }
        let mut lts = TransitionSystem { adj };
        lts.sort_labels();
        lts
    }

    pub fn sort_labels(&mut self) {
        for row in self.adj.iter_mut() {
            row.sort_by_key(|t| t.label);
        }
    }

    pub fn n_vertices(&self) -> u32 {
        self.adj.len() as u32
    }

    pub fn build_game_graph(self, p: u32, q: u32) -> GameGraph {
        let mut builder = GameBuild::with_lts(self);
        builder.build(p, q);
        builder.game
    }

    pub async fn winning_budgets(self, p: u32, q: u32) -> Result<Vec<EnergyArray>> {
        let game_graph = self.build_game_graph(p, q);
        let mut game = EnergyGame::standard_reach(game_graph);
        game.run().await?;
        Ok(game.energies)
    }
}
