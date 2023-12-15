pub mod gamepos;

use std::collections::HashMap;
use std::rc::Rc;

use crate::error::Result;
use crate::energy::EnergyConf;
use crate::GameGraph;

use gamepos::*;

pub struct TransitionSystem {
    pub adj: Vec<Vec<u32>>,
    // Transition labels encoded as i32:
    // 0 => τ
    // k => Channel index k, k ∈ ℕ
    // -k => Co-Action of k, k ∈ ℕ
    //
    // Actual names (Strings) should be stored in a separate list
    pub labels: Vec<Vec<i32>>,
}

impl TransitionSystem {
    pub fn new(n_vertices: u32, edges: Vec<(u32, u32, i32)>) -> Self {
        let mut adj = vec![vec![]; n_vertices as usize];
        let mut labels = vec![vec![]; n_vertices as usize];
        for (from, to, label) in edges {
            adj[from as usize].push(to);
            labels[from as usize].push(label);
        }
        TransitionSystem {
            adj,
            labels,
        }
    }

    pub fn n_vertices(&self) -> u32 {
        self.adj.len() as u32
    }
}

pub struct GameBuild {
    lts: TransitionSystem,
    game: GameGraph,
    nodes: Vec<Rc<Position>>,
    node_map: HashMap<Rc<Position>, usize>,
}

impl GameBuild {
    pub async fn with_lts(lts: TransitionSystem) -> Result<Self> {
        Ok(GameBuild {
            lts,
            game: GameGraph::empty(EnergyConf::STANDARD),
            nodes: Vec::new(),
            node_map: HashMap::new(),
        })
    }
}
