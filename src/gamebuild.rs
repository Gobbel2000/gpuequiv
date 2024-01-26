pub mod gamepos;

use std::cmp::Ordering;
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};
use ndarray::Axis;

use crate::TransitionSystem;
use crate::energy::{EnergyConf, UpdateArray};
use crate::energygame::GameGraph;

pub use gamepos::*;

impl TransitionSystem {
    fn compare_enabled(&self, p: u32, q: u32) -> (bool, bool) {
        if p == q {
            return (true, true);
        }
        let mut subset = true;
        let mut superset = true;

        let p_labels = &self.adj[p as usize];
        let q_labels = &self.adj[q as usize];

        let mut p_ptr = 0;
        let p_max = p_labels.len();
        let mut q_ptr = 0;
        let q_max = q_labels.len();
        loop {
            if !(subset || superset) {
                // Already incomparable
                break;
            }

            if p_ptr == p_max && q_ptr == q_max {
                // Both lists exhausted
                break;
            }
            if p_ptr == p_max {
                // Exhausted p, q has additional members, so is not a subset of p
                subset = false;
                break;
            }
            if q_ptr == q_max {
                // Exhausted q, p has additional members. q is not a superset of p
                superset = false;
                break;
            }

            let p_act = p_labels[p_ptr].label;
            let q_act = q_labels[q_ptr].label;

            let mut advance_p = false;
            let mut advance_q = false;

            match p_act.cmp(&q_act) {
                Ordering::Equal => {
                    // Both p and q have this action, continue
                    advance_p = true;
                    advance_q = true;
                },
                Ordering::Less => {
                    // p has an action that q doesn't have
                    superset = false;
                    advance_p = true;
                },
                Ordering::Greater => {
                    // q has an action that p doesn't have
                    subset = false;
                    advance_q = true;
                },
            }

            if advance_p {
                // Skip ahead until next distinct action, or end of list, but
                // always at least by one.
                p_ptr += 1;
                while p_ptr < p_max && p_labels[p_ptr].label == p_act {
                    p_ptr += 1;
                }
            }
            if advance_q {
                q_ptr += 1;
                while q_ptr < q_max && q_labels[q_ptr].label == q_act {
                    q_ptr += 1;
                }
            }
        }
        (subset, superset)
    }
}


pub struct GameBuild {
    pub lts: TransitionSystem,
    pub game: GameGraph,
    pub nodes: Vec<Rc<Position>>,
    pub node_map: FxHashMap<Rc<Position>, u32>,
}

impl GameBuild {
    const ENERGY_CONF: EnergyConf = EnergyConf::STANDARD;

    pub fn with_lts(lts: TransitionSystem) -> Self {
        GameBuild {
            lts,
            game: GameGraph::empty(Self::ENERGY_CONF),
            nodes: Vec::new(),
            node_map: FxHashMap::default(),
        }
    }

    pub fn compare(&mut self, p: u32, q: u32) {
        self.new_node(Position::attack(p, vec![q]));
        self.build_internal();
    }

    pub fn compare_all(&mut self) -> u32 {
        for p in 0..self.lts.n_vertices() {
            for q in 0..p {
                // Only compare if p and q have the same enabled actions
                if self.lts.compare_enabled(p, q) == (true, true) {
                    self.new_node(Position::attack(p, vec![q]));
                }
            }
        }
        let n_starting_points = self.game.n_vertices();
        self.build_internal();
        n_starting_points
    }

    pub fn compare_all_but_bisimilar(&mut self) -> u32 {
        // Compute bisimulation
        let partition = self.lts.signature_refinement();
        // Pick one representative for each bisimulation equivalence class
        let mut represented = FxHashSet::default();
        let mut representatives = Vec::new();
        for (proc, part) in partition.iter().enumerate() {
            if !represented.contains(part) {
                representatives.push(proc as u32);
                represented.insert(part);
            }
        }
        // Compare all representatives with each other
        for (i, &p) in representatives.iter().enumerate() {
            for &q in &representatives[..i] {
                self.new_node(Position::attack(p, vec![q]));
            }
        }
        let n_starting_points = self.game.n_vertices();
        self.build_internal();
        n_starting_points
    }

    fn build_internal(&mut self) {
        let mut idx = 0;
        while idx < self.game.n_vertices() {
            let (positions, weights) = self.nodes[idx as usize].successors(&self.lts);
            self.add_nodes(positions, weights);
            idx += 1;
        }
        self.game.row_offsets.push(self.game.column_indices.len() as u32);
        self.game.make_reverse();
    }

    // Assumes that `positions` contains no duplicates
    fn add_nodes(&mut self, positions: Vec<Position>, weights: UpdateArray) {
        // Start range of next node
        self.game.row_offsets.push(self.game.column_indices.len() as u32);
        for pos in positions {
            // Gather indices of provided positions
            let suc = self.node_map.get(&pos).copied()  // Existing node
                .unwrap_or_else(|| self.new_node(pos)); // New node
            self.game.column_indices.push(suc);
        }
        self.game.weights.array.append(Axis(0), weights.array.view()).unwrap();
    }

    // Create new node and return its index
    fn new_node(&mut self, pos: Position) -> u32 {
        let idx = self.game.n_vertices();
        // Add new empty node into game graph
        self.game.attacker_pos.push(pos.is_attack());

        // Keep track of  positions <=> idx  symbol tables
        let pos_rc = Rc::new(pos);
        self.nodes.push(Rc::clone(&pos_rc));
        self.node_map.insert(Rc::clone(&pos_rc), idx);
        idx
    }
}
