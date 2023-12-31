pub mod gamepos;

use std::collections::VecDeque;
use std::cmp::Ordering;
use std::ops::Range;
use std::rc::Rc;

use rustc_hash::FxHashMap;

use crate::TransitionSystem;
use crate::energy::{EnergyConf, UpdateArray};
use crate::energygame::{GameGraph, make_reverse};

use gamepos::*;

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
    lts: TransitionSystem,
    pub game: GameGraph,
    pub nodes: Vec<Rc<Position>>,
    node_map: FxHashMap<Rc<Position>, u32>,
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

    pub fn build(&mut self, p: u32, q: u32) {
        let idx = self.new_node(Position::Attack(AttackPosition {
            p,
            q: vec![q],
        }));
        let mut frontier = VecDeque::from([idx]);
        while let Some(idx) = frontier.pop_front() {
            let (positions, weights) = self.nodes[idx as usize].successors(&self.lts);
            let new_indices = self.add_nodes(idx as usize, positions, weights);
            frontier.extend(new_indices);
        }
        self.game.reverse = make_reverse(&self.game.adj);
    }

    // Assumes that `positions` contains no duplicates
    fn add_nodes(
        &mut self,
        start: usize,
        positions: Vec<Position>,
        weights: UpdateArray,
    ) -> Range<u32> {
        let first_new = self.game.adj.len() as u32;
        // Gather indices of provided positions
        self.game.adj[start] = positions.into_iter().map(|pos|
                self.node_map.get(&pos).copied()           // Existing node
                    .unwrap_or_else(|| self.new_node(pos)) // New node
            )
            .collect();
        self.game.weights[start] = weights;
        // Return slice of newly added node indices
        first_new..self.game.adj.len() as u32
    }

    // Create new node and return its index
    fn new_node(&mut self, pos: Position) -> u32 {
        let idx = self.game.adj.len() as u32;
        // Add new empty node into game graph
        self.game.adj.push(Vec::new());
        self.game.weights.push(UpdateArray::empty(Self::ENERGY_CONF));
        self.game.attacker_pos.push(pos.is_attack());

        // Keep track of  positions <=> idx  symbol tables
        let pos_rc = Rc::new(pos);
        self.nodes.push(Rc::clone(&pos_rc));
        self.node_map.insert(Rc::clone(&pos_rc), idx);
        idx
    }
}
