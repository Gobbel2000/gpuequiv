use gpuequiv::*;

fn example_graph() -> GameGraph {
    /*
    let attacker_pos: Vec<bool> = (0..18)
        .map(|i| [0, 2, 4, 6, 9, 11, 12, 14, 17].contains(&i))
        .collect();
        */
    let attacker_pos = vec![true; 18];
    GameGraph::new(
        18,
        &[
            (0, 1, 0.into()),
            (1, 2, 0.into()),
            (2, 3, 0.into()),
            (3, 4, 0.into()),
            (4, 5, 0.into()),
            (5, 6, 0.into()),
            (6, 7, (3 << 8).into()),
            (6, 8, 0.into()),
            (8, 5, 0.into()),
            (1, 9, 0.into()),
            (9, 10, 0.into()),
            (10, 11, 0.into()),
            (11, 3, 0.into()),
            (10, 12, 0.into()),
            (12, 10, 0.into()),
            (0, 13, 0.into()),
            (13, 14, 0.into()),
            (14, 15, 0.into()),
            (14, 16, 0.into()),
            (16, 17, 0.into()),
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
    game.energies[7] = vec![0x1.into(), 0x3.into(), 0x30.into()];
    println!("{:#?}", game);
    let mut runner = game.get_gpu_runner().await;
    runner.execute_gpu().await.unwrap();
}

fn main() {
    pollster::block_on(execute_reachability_small());
}

