use gpuequiv::*;

fn example_graph() -> GameGraph {
    let attacker_pos = vec![true; 18];
    GameGraph::new(
        18,
        &[
            (0, 1, Update::zero()),
            (1, 2, Update::zero()),
            (2, 3, Update::zero()),
            (3, 4, Update::zero()),
            (4, 5, Update::zero()),
            (5, 6, Update::zero()),
            (6, 7, update![Upd::None, Upd::None, Upd::Min(1)]),
            (6, 8, Update::zero()),
            (8, 5, Update::zero()),
            (1, 9, Update::zero()),
            (9, 10, Update::zero()),
            (10, 11, Update::zero()),
            (11, 3, Update::zero()),
            (10, 12, Update::zero()),
            (12, 10, Update::zero()),
            (0, 13, Update::zero()),
            (13, 14, Update::zero()),
            (14, 15, Update::zero()),
            (14, 16, Update::zero()),
            (16, 17, Update::zero()),
        ],
        &attacker_pos,
    )
}

async fn execute_reachability_small() {
    let mut game = EnergyGame {
        graph: example_graph(),
        energies: vec![vec![]; 18],
        to_reach: vec![6, 15],
    };
    game.energies[7] = vec![energy![1], energy![3], energy![0, 0, 3], energy![]];
    //println!("{:#?}", game);
    let mut runner = game.get_gpu_runner().await;
    runner.execute_gpu().await.unwrap();
}

fn main() {
    pollster::block_on(execute_reachability_small());
}

