use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttackPosition {
    pub p: u32,
    pub q: Vec<u32>,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DefendPosition {
    pub p: u32,
    pub q: Vec<u32>,
    pub qx: Vec<u32>,
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
