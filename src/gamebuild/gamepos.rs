use std::fmt;
use std::sync::OnceLock;

use ndarray::aview1;
use crate::energy::{UpdateArray, Update, Upd, FromEnergyConf};
use crate::update;
use super::{GameBuild, TransitionSystem};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttackPosition {
    pub p: u32,
    pub q: Vec<u32>,
}

impl AttackPosition {
    fn successors(&self, lts: &TransitionSystem) -> (Vec<Position>, UpdateArray) {
        let (observation_update, challenge_update) = {
            static ONCE: OnceLock<(Update, Update)> = OnceLock::new();
            ONCE.get_or_init(|| (
                Update::new(&update![Upd::Decrement], GameBuild::ENERGY_CONF).unwrap(),
                Update::new(&update![0, Upd::Decrement], GameBuild::ENERGY_CONF).unwrap(),
            ))
        };
        let mut positions = Vec::with_capacity(lts.adj[self.p as usize].len() + 4);

        // Observation moves
        for transition in &lts.adj[self.p as usize] {
            positions.push(Position::attack(
                transition.process,
                self.set_transition(lts, transition.label)
            ));
        }
        let mut weights = UpdateArray::repeat(observation_update.clone(), positions.len());

        // Conjunction challenges
        let conjunctions_start = positions.len();
        let p = self.p;
        // Q* = ∅
        positions.push(Position::defend(p, self.q.clone(), Vec::new()));

        let mut q_subset = Vec::new();
        let mut qx_subset = Vec::new();
        let mut q_superset = Vec::new();
        let mut qx_superset = Vec::new();
        let mut q_equal = Vec::new();
        let mut qx_equal = Vec::new();

        for &q in &self.q {
            let (subset, superset) = lts.compare_enabled(p, q);
            if subset {
                qx_subset.push(q);
            } else {
                q_subset.push(q);
            }
            if superset {
                qx_superset.push(q);
            } else {
                q_superset.push(q);
            }
            if subset && superset {
                qx_equal.push(q);
            } else {
                q_equal.push(q);
            }
        }

        // Check for duplicates before inserting each position.
        // This is done via linear search which is fine, since these are at most 4 positions
        // (the empty set position added above must also be considered).
        let subset_pos = Position::defend(p, q_subset, qx_subset);
        if !positions[conjunctions_start..].contains(&subset_pos) {
            positions.push(subset_pos);
        }
        let superset_pos = Position::defend(p, q_superset, qx_superset);
        if !positions[conjunctions_start..].contains(&superset_pos) {
            positions.push(superset_pos);
        }
        let equal_pos = Position::defend(p, q_equal, qx_equal);
        if !positions[conjunctions_start..].contains(&equal_pos) {
            positions.push(equal_pos);
        }

        weights.push_n(challenge_update.clone(), positions.len() - conjunctions_start);

        (positions, weights)
    }

    // Calculate  Q -a-> Q'
    // That is,  q' ∈ Q'  iff  ∃q ∈ Q . q -a-> q'
    fn set_transition(&self, lts: &TransitionSystem, action: i32) -> Vec<u32> {
        let mut q: Vec<u32> = self.q.iter().flat_map(|&q| lts.adj[q as usize].iter()
                .filter(|t| t.label == action)
                .map(|t| t.process)
            )
            .collect();
        q.sort_unstable();
        q.dedup();
        q
    }
}

impl fmt::Display for AttackPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attacker Position: p = {},\t Q = {:?}",
               self.p, self.q,
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SingletonPosition {
    pub p: u32,
    pub q: u32,
}

impl SingletonPosition {
    fn successors(&self) -> (Vec<Position>, UpdateArray) {
        // Save both possible update arrays in static memory
        let (single_update_array, both_update_array) = {
            static ONCE: OnceLock<(UpdateArray, UpdateArray)> = OnceLock::new();
            ONCE.get_or_init(|| (
                UpdateArray::from_conf([
                        update![Upd::Min(4)]
                    ].as_slice(),
                    GameBuild::ENERGY_CONF).unwrap(),
                UpdateArray::from_conf([
                        update![Upd::Min(4)],
                        update![Upd::Min(5), 0, 0, 0, 0, Upd::Decrement]
                    ].as_slice(),
                    GameBuild::ENERGY_CONF).unwrap(),
            ))
        };

        let positive = Position::attack(self.p, vec![self.q]);
        if self.p == self.q {
            (vec![positive], single_update_array.clone())
        } else {
            // Negative decision: Swap p and q
            (vec![positive, Position::attack(self.q, vec![self.p])],
             both_update_array.clone())
        }
    }
}

impl fmt::Display for SingletonPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attacker Clause Position: p = {},\t q = {}",
               self.p, self.q,
        )
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DefendPosition {
    pub p: u32,
    pub q: Vec<u32>,
    pub qx: Vec<u32>,
}

impl DefendPosition {
    fn successors(&self) -> (Vec<Position>, UpdateArray) {
        let (revival_update, answer_update) = {
            static ONCE: OnceLock<(Update, Update)> = OnceLock::new();
            ONCE.get_or_init(|| (
                Update::new(&update![Upd::Min(3)], GameBuild::ENERGY_CONF).unwrap(),
                Update::new(&update![0, 0, 0, Upd::Min(3)], GameBuild::ENERGY_CONF).unwrap(),
            ))
        };
        let p = self.p;
        let mut positions = Vec::with_capacity(self.q.len() + (!self.qx.is_empty()) as usize);
        positions.extend(self.q.iter().map(|&q| Position::clause(p, q)));
        let mut weights = UpdateArray::repeat(answer_update.clone(), positions.len());

        if !self.qx.is_empty() {
            positions.push(Position::attack(p, self.qx.clone()));
            weights.array.push_row(aview1(&revival_update.data)).unwrap();
        }

        (positions, weights)
    }
}

impl fmt::Display for DefendPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Defender Conjunction Position: p = {},\t Q = {:?},\t Q* = {:?}",
               self.p, self.q, self.qx,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Position {
    Attack(AttackPosition),
    Clause(SingletonPosition),
    Defend(DefendPosition),
}

impl Position {
    pub fn attack(p: u32, q: Vec<u32>) -> Self {
        Position::Attack(AttackPosition { p, q })
    }

    pub fn clause(p: u32, q: u32) -> Self {
        Position::Clause(SingletonPosition { p, q })
    }

    pub fn defend(p: u32, q: Vec<u32>, qx: Vec<u32>) -> Self {
        Position::Defend(DefendPosition { p, q, qx })
    }

    pub fn is_attack(&self) -> bool {
        matches!(self, Position::Attack(_) | Position::Clause(_))
    }

    pub fn successors(&self, lts: &TransitionSystem) -> (Vec<Position>, UpdateArray) {
        match self {
            Position::Attack(p) => p.successors(lts),
            Position::Clause(p) => p.successors(),
            Position::Defend(p) => p.successors(),
        }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Position::Attack(p) => write!(f, "{p}"),
            Position::Clause(p) => write!(f, "{p}"),
            Position::Defend(p) => write!(f, "{p}"),
        }
    }
}
