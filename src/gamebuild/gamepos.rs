use std::fmt;

const LIST_CHUNK_SIZE: usize = 4;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct LinkedList {
    pub(crate) data: [u32; LIST_CHUNK_SIZE],
    pub(crate) len: u32,
    pub(crate) next: u32,
}

impl LinkedList {
    fn to_vec(&self, heap: &[LinkedList]) -> Vec<u32> {
        let mut len = self.len as i32;
        let mut vec = Vec::with_capacity(len as usize);
        let chunk_len = LIST_CHUNK_SIZE.min(len as usize);
        vec.extend_from_slice(&self.data[..chunk_len]);

        len -= LIST_CHUNK_SIZE as i32;
        let mut next = self.next;
        while len > 0 {
            let chunk = heap[next as usize];
            let chunk_len = LIST_CHUNK_SIZE.min(len as usize);
            vec.extend_from_slice(&chunk.data[..chunk_len]);

            len -= LIST_CHUNK_SIZE as i32;
            next = chunk.next;
        }
        vec
    }
}

impl fmt::Display for LinkedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..LIST_CHUNK_SIZE.min(self.len as usize) {
            if i == 0 {
                write!(f, "{}", self.data[i as usize])?;
            } else {
                write!(f, ", {}", self.data[i as usize])?;
            }
        }
        write!(f, "]")?;
        if self.len as usize > LIST_CHUNK_SIZE {
            write!(f, "->{}", self.next)?;
        }
        Ok(())
    }
}

// Enable bytemucking for filling buffers
unsafe impl bytemuck::Zeroable for LinkedList {}
unsafe impl bytemuck::Pod for LinkedList {}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct AttackPosition {
    pub(crate) p: u32,
    pub(crate) q: LinkedList,
}

impl fmt::Display for AttackPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attacker Position: p = {},\t Q = {}",
               self.p, self.q,
        )
    }
}

unsafe impl bytemuck::Zeroable for AttackPosition {}
unsafe impl bytemuck::Pod for AttackPosition {}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SingletonPosition {
    pub p: u32,
    pub q: u32,
}

unsafe impl bytemuck::Zeroable for SingletonPosition {}
unsafe impl bytemuck::Pod for SingletonPosition {}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct DefendPosition {
    p: u32,
    q: LinkedList,
    qx: LinkedList,
}

impl fmt::Display for DefendPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Defender Conjunction Position: p = {},\t Q = {},\t Q* = {}",
               self.p, self.q, self.qx,
        )
    }
}

unsafe impl bytemuck::Zeroable for DefendPosition {}
unsafe impl bytemuck::Pod for DefendPosition {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Metadata {
    pub(crate) heap_top: u32,
    pub(crate) heap_oom: u32,
}

unsafe impl bytemuck::Zeroable for Metadata {}
unsafe impl bytemuck::Pod for Metadata {}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Position {
    Attack {
        p: u32,
        q: Vec<u32>,
    },
    Clause(SingletonPosition),
    Defend {
        p: u32,
        q: Vec<u32>,
        qx: Vec<u32>,
    },
}

impl Position {
    pub(crate) fn attack(pos: AttackPosition, heap: &[LinkedList]) -> Self {
        let mut q = pos.q.to_vec(heap);
        q.sort();
        Position::Attack {
            p: pos.p,
            q,
        }
    }

    pub(crate) fn attack_singleton(pos: SingletonPosition) -> Self {
        Position::Attack {
            p: pos.p,
            q: vec![pos.q],
        }
    }

    pub(crate) fn clause(pos: SingletonPosition) -> Self {
        Position::Clause(pos)
    }

    pub(crate) fn defend(pos: DefendPosition, heap: &[LinkedList]) -> Self {
        let mut q = pos.q.to_vec(heap);
        let mut qx = pos.qx.to_vec(heap);
        q.sort();
        qx.sort();
        Position::Defend {
            p: pos.p,
            q,
            qx,
        }
    }
}
