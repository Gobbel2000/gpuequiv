pub mod gamepos;

use std::cmp::Ordering;
use std::ops::Deref;
use std::rc::Rc;

use rustc_hash::FxHashMap;
use ndarray::Axis;

use crate::{TransitionSystem, StartInfo};
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


#[derive(Clone, Debug)]
pub struct GameBuild {
    pub game: GameGraph,
    pub nodes: Vec<Rc<Position>>,
    pub node_map: FxHashMap<Rc<Position>, u32>,
    pub n_starting_points: usize,
}

impl GameBuild {
    const ENERGY_CONF: EnergyConf = EnergyConf::STANDARD;

    pub fn new() -> Self {
        GameBuild {
            game: GameGraph::empty(Self::ENERGY_CONF),
            nodes: Vec::new(),
            node_map: FxHashMap::default(),
            n_starting_points: 0,
        }
    }

    pub fn compare(lts: &TransitionSystem, p: u32, q: u32) -> Self {
        let mut builder = Self::default();
        builder.new_node(Position::attack(p, vec![q]));
        builder.build_internal(lts);
        builder
    }

    pub fn compare_multiple(lts: &TransitionSystem, comparisons: &[(u32, u32)]) -> Self {
        let mut builder = Self::default();
        for (p, q) in comparisons {
            builder.new_node(Position::attack(*p, vec![*q]));
        }
        builder.build_internal(lts);
        builder
    }

    pub fn compare_all(lts: &TransitionSystem) -> (Self, StartInfo) {
        let mut builder = Self::default();
        for p in 0..lts.n_vertices() {
            for q in 0..lts.n_vertices() {
                // Only compare if p and q have the same enabled actions
                if p != q && lts.compare_enabled(p, q) == (true, true) {
                    builder.new_node(Position::attack(p, vec![q]));
                }
            }
        }
        builder.build_internal(lts);
        let start_info = StartInfo::new(builder.starting_points(), lts.n_vertices() as usize);
        (builder, start_info)
    }

    /// **Do not use this function, instead minimize the LTS with `lts.bisimilar_minimize`,
    /// and then use `compare_all`.
    #[doc(hidden)]
    pub fn compare_all_but_bisimilar(lts: &TransitionSystem) -> (Self, StartInfo) {
        // Compute bisimulation
        let (partition, count) = lts.signature_refinement();
        // Pick one representative for each bisimulation equivalence class
        let mut represented = vec![false; count];
        let mut representatives = Vec::new();
        for (proc, part) in partition.iter().enumerate() {
            if !represented[*part] {
                representatives.push(proc as u32);
                represented[*part] = true;
            }
        }
        // Compare all representatives with each other
        let mut builder = Self::default();
        for &p in &representatives {
            for &q in &representatives {
                // Only compare if p and q have the same enabled actions
                if p != q && lts.compare_enabled(p, q) == (true, true) {
                    builder.new_node(Position::attack(p, vec![q]));
                }
            }
        }
        builder.build_internal(lts);
        let start_info = StartInfo::from_partition(builder.starting_points(), &partition);
        (builder, start_info)
    }

    pub fn starting_points(&self) -> Vec<AttackPosition> {
        self.nodes.iter()
            .take(self.n_starting_points)
            .filter_map(|pos| match Deref::deref(pos) {
                // Normally all starting positions should be attack positions
                Position::Attack(p) => Some(p.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn take_game(&mut self) -> GameGraph {
        std::mem::replace(&mut self.game, GameGraph::empty(Self::ENERGY_CONF))
    }

    fn build_internal(&mut self, lts: &TransitionSystem) {
        self.n_starting_points = self.game.n_vertices() as usize;
        let mut idx = 0;
        while idx < self.game.n_vertices() {
            let (positions, weights) = self.nodes[idx as usize].successors(lts);
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

impl Default for GameBuild {
    fn default() -> Self {
        GameBuild::new()
    }
}
