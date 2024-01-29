// Build a game graph from an LTS (Labeled Transition System).
// The LTS is taken from:
// Bisping 2023 - Process Equivalence Problems as Energy Games, Figure 4

use gpuequiv::*;
use gpuequiv::energygame::*;

fn main() {
    let lts = TransitionSystem::new(
        3,
        vec![
            (0, 0, 0),
            (0, 2, 0),
            (0, 2, 1),
            (1, 1, 0),
            (1, 2, 1),
            (2, 2, 0),
        ],
    );
    let mut builder = gamebuild::GameBuild::with_lts(lts);
    // Build game graph that can compare processes 0 and 1.
    // This initializes the graph with the Attack node (0, {1})a
    builder.compare(0, 1);

    println!("Nodes:");
    for (i, p) in builder.nodes.iter().enumerate() {
        println!("{i}: \t{p}");
    }
    // Less optimized graph representation, exists for serialization purposes
    let nicer_graph = SerdeGameGraph::from(builder.game);
    println!("\nGraph:");
    println!("conf: {:?}", nicer_graph.conf);
    println!("adj: {:?}", nicer_graph.adj);
    println!("weights: {:?}", nicer_graph.weights);
    println!("attacker_pos: {:?}", nicer_graph.attacker_pos);
}
