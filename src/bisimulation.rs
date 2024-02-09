use rustc_hash::FxHashMap;

use super::{TransitionSystem, Transition};

impl TransitionSystem {

    /// Find strong bisimulation using signature refinement.
    /// The first return value is the partition, the second the number of bisimilarity classes.
    ///
    /// The partition list contains an index of the bisimulation equivalence class for each process.
    /// That is, if `p = signature_refinement()`, then `p[i] == p[j]` iff `i ~ j`.
    /// In other words, if and only if two processes get assigned the same partition index,
    /// they are bisimilar.
    ///
    /// This sequential algorithm is described by S. Blom and S. Orzan in
    /// "A Distributed Algorithm for Strong Bisimulation Reduction of State Spaces", 2002.
    pub fn signature_refinement(&self) -> (Vec<usize>, usize) {
        let mut partition = vec![0; self.n_vertices() as usize];
        let mut prev_count: usize = 0;
        let mut new_count: usize = 1;
        while prev_count != new_count {
            prev_count = new_count;
            new_count = 0;
            let signatures = self.signatures(&partition);
            let mut sigmap = FxHashMap::default();
            for sig in &signatures {
                sigmap.entry(sig).or_insert_with(|| {
                    new_count += 1;
                    new_count - 1
                });
            }
            for (part, sig) in partition.iter_mut().zip(&signatures) {
                *part = sigmap[&sig];
            }
        }
        (partition, new_count)
    }

    // Returns a set-valued signature for each process i:
    // sig[i] = {(a, ID) | i -a-> j and partition[j] == ID}
    fn signatures(&self, partition: &[usize]) -> Vec<Vec<(i32, usize)>> {
        self.adj.iter().map(|adj| {
                let mut sig: Vec<_> = adj.iter()
                    .map(|transition| (transition.label, partition[transition.process as usize]))
                    .collect();
                sig.sort_unstable();
                sig.dedup();
                sig
            })
            .collect()
    }

    pub fn bisimilar_minimize(&self) -> (TransitionSystem, Vec<usize>) {
        let (bisim, count) = self.signature_refinement();
        let mut adj = vec![vec![]; count];
        let mut represented = vec![false; count];
        for (process, &partition) in bisim.iter().enumerate() {
            if !represented[partition] {
                if adj.len() <= partition {
                    adj.resize(partition + 1, Vec::new());
                }
                adj[partition] = self.adj[process].iter()
                    .map(|transition| Transition {
                        process: bisim[transition.process as usize] as u32,
                        label: transition.label,
                    })
                    .collect();
                represented[partition] = true;
            }
        }
        (adj.into(), bisim)
    }
}
