use rustc_hash::FxHashMap;

use super::TransitionSystem;

impl TransitionSystem {

    // Find strong bisimulation using signature refinement.
    // Returns a list with the index of the bisimulation equivalence class for each process.
    // That is, if p = signature_refinement(), then p[i] == p[j] iff i ~ j.
    // In other words, if and only if two processes get assigned the same partition index,
    // they are bisimilar.
    //
    // This sequential algorithm is described by S. Blom and S. Orzan in
    // "A Distributed Algorithm for Strong Bisimulation Reduction of State Spaces", 2002.
    pub fn signature_refinement(&self) -> Vec<u32> {
        let mut partition = vec![0; self.n_vertices() as usize];
        let mut prev_count: u32 = 0;
        let mut new_count: u32 = 1;
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
        partition
    }

    // Returns a set-valued signature for each process i:
    // sig[i] = {(a, ID) | i -a-> j and partition[j] == ID}
    fn signatures(&self, partition: &[u32]) -> Vec<Vec<(i32, u32)>> {
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
}
