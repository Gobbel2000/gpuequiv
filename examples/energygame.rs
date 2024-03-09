// In this example an energy game is manually constructed and then solved on the GPU.
// The resulting winning energies are printed to stdout.

use gpuequiv::*;
use gpuequiv::energygame::*;

fn game() -> EnergyGame {
    let conf = EnergyConf::STANDARD;
    // Indicates which node is an attacker node
    let attacker_pos: Vec<bool> = (0..20)
        .map(|i| [1, 2, 3, 4, 6, 10, 16].contains(&i))
        .collect();
    let graph = GameGraph::new(
        20, // Number of nodes
        vec![
            (0, 1, update![-1]), // Edge with start node, target node and weight (update)
            (0, 2, update![0, -1]),
            (0, 3, update![0, 0, 0, -1]),
            (0, 4, update![0, 0, 0, 0, 0, -1]),

            (1, 5, update![-1]),
            (5, 6, update![-1]),
            (6, 7, update![0, -1]),
            (6, 8, update![-1, 0]),
            (5, 10, update![Upd::Min(3)]),
            (1, 9, update![0, 0, -1]),
            (9, 10, update![0, 0, -1]),
            (10, 11, update![-1]),

            (2, 12, update![0, 0, -1]),
            (2, 13, update![0, -1]),

            (3, 14, update![0, 0, -1]),
            (3, 15, update![0, 0, 0, Upd::Min(5)]),
            (15, 16, update![0, 0, 0, -1]),
            (16, 17, update![0, 0, 0, -1]),
            (3, 18, update![0, -1]),

            (4, 19, update![0, 0, 0, 0, 0, -1]),
        ],
        attacker_pos,
        conf,
    );
    EnergyGame::standard_reach(graph)
}

async fn run_game() -> Result<()> {
    let game = game();

    // Run algorithm on GPU
    let energies = game.run().await?;

    for node in energies {
        println!("{}", node);
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    pollster::block_on(run_game())
}
