use std::time::Instant;
use std::collections::HashMap;

use gpuequiv::challenge_wgpu::TransitionSystem;


#[derive(Debug, Clone)]
struct AtkPos {
    p: u32,
    q: Vec<u32>,
}

fn compare_enabled(lts: &TransitionSystem, p: u32, q: u32) -> (bool, bool) {
    if p == q {
        return (true, true);
    }
    let mut subset = true;
    let mut superset = true;

    let p_labels = &lts.labels[p as usize];
    let q_labels = &lts.labels[q as usize];

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

        let p_act = p_labels[p_ptr];
        let q_act = q_labels[q_ptr];

        let mut advance_p = false;
        let mut advance_q = false;

        if p_act == q_act {
            // Both p and q have this action, continue
            advance_p = true;
            advance_q = true;
        } else if p_act < q_act {
            // p has an action that q doesn't have
            superset = false;
            advance_p = true;
        } else if q_act < p_act {
            // q has an action that p doesn't have
            subset = false;
            advance_q = true;
        }

        if advance_p {
            // Skip ahead until next distinct action, or end of list, but
            // always at least by one.
            p_ptr += 1;
            while p_ptr < p_max && p_labels[p_ptr] == p_act {
                p_ptr += 1;
            }
        }
        if advance_q {
            q_ptr += 1;
            while q_ptr < q_max && q_labels[q_ptr] == q_act {
                q_ptr += 1;
            }
        }
    }
    (subset, superset)
}

impl AtkPos {
    fn process_challenge(&self, lts: &TransitionSystem) -> [DefPos; 4] {
        let p = self.p;
        let empty = DefPos {
            p,
            q: self.q.clone(),
            qx: Vec::new(),
        };

        let mut q_subset = Vec::new();
        let mut qx_subset = Vec::new();
        let mut q_superset = Vec::new();
        let mut qx_superset = Vec::new();
        let mut q_equal = Vec::new();
        let mut qx_equal = Vec::new();

        for &q in &self.q {
            let (subset, superset) = compare_enabled(lts, p, q);
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
        
        [
            empty,
            DefPos {            
                p,
                q: q_subset,
                qx: qx_subset,
            },
            DefPos {
                p,
                q: q_superset,
                qx: qx_superset,
            },
            DefPos {
                p,
                q: q_equal,
                qx: qx_equal,
            },
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DefPos {
    p: u32,
    q: Vec<u32>,
    qx: Vec<u32>,
}

fn main() {
    let lts = TransitionSystem::new(
        11,
        vec![
            (0, 1, 1),
            (0, 2, 1),
            (0, 2, 2),
            (0, 1, 3),
            (0, 2, 3),
            (0, 2, 4),
            (0, 1, 5),
            (0, 2, 5),

            (1, 3, 1),
            (1, 3, 2),
            (1, 3, 3),

            (2, 3, 1),
            (2, 3, 2),
            (2, 3, 3),

            (3, 1, 2),

            (4, 0, 4),
            (5, 0, 4),
            (6, 0, 4),

            (7, 1, 1),
            (8, 1, 1),
            (9, 1, 1),

            (10, 3, 1),
            (10, 3, 2),
            (10, 3, 3),
            (10, 3, 4),
        ],
    );


    let mut positions = vec![
        AtkPos {
            p: 2,
            q: vec![10, 3],
        },
        AtkPos {
            p: 1,
            q: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        AtkPos {
            p: 0,
            q: vec![0, 4, 5, 6, 8, 9, 10],
        },
        AtkPos {
            p: 3,
            q: vec![10, 7, 8, 9],
        },
        AtkPos {
            p: 2,
            q: vec![3],
        },
    ];

    for _ in 0..1024 {
        positions.extend_from_within(..5);
    }

    let now = Instant::now();
    for _ in 0..1000 {
        let mut pos_out = Vec::with_capacity(positions.len() * 4);
        let mut map = HashMap::new();
        let mut n = 0;
        for pos in &positions {
            for p in pos.process_challenge(&lts) {
                if !map.contains_key(&p) {
                    map.insert(p.clone(), n); 
                    pos_out.push(p);
                    n += 1;
                }
            }
        }

        if map.contains_key(&DefPos { p: 99, q: Vec::new(), qx: Vec::new() }) {
            panic!("Huh");
        }
    }
    let elapsed = now.elapsed();
    println!("Took {}ms", elapsed.as_millis());

    /*
    for p in pos_out[..100].iter() {
        println!("{:?}", p);
    }
    */
}
