use gpuequiv::*;

fn example_graph() -> GameGraph {
    let attacker_pos = vec![true; 18];
    GameGraph::new(
        18,
        &[
            (0, 1, update![Upd::Decrement]),
            (1, 2, update![Upd::Decrement]),
            (2, 3, update![Upd::Decrement]),
            (3, 4, update![Upd::Decrement]),
            (4, 5, update![Upd::Decrement]),
            (5, 6, update![Upd::Decrement]),
            (6, 7, update![Upd::Decrement]),
            (6, 8, update![Upd::Decrement]),
            (8, 4, update![Upd::Decrement]),
            (1, 9, update![Upd::Decrement]),
            (9, 10, update![Upd::Decrement]),
            (10, 11, update![Upd::Decrement]),
            (11, 3, update![Upd::Decrement]),
            (10, 12, update![Upd::Decrement]),
            (12, 10, update![Upd::Decrement]),
            (0, 13, update![Upd::Decrement]),
            (13, 14, update![Upd::Decrement]),
            (14, 15, update![Upd::Decrement]),
            (14, 16, update![Upd::Decrement]),
            (16, 17, update![Upd::Decrement]),
        ],
        &attacker_pos,
    )
}

async fn execute_reachability_small() {
    /*
    let mut game = EnergyGame {
        graph: example_graph(),
        energies: vec![vec![]; 18],
        to_reach: vec![7, 15],
    };
    */
    let mut game = EnergyGame::from_graph(example_graph())
        .with_reach(vec![6, 15]);
    /*
    game.energies[7] = vec![
        energy![],
        energy![],
        energy![],
    ];
    game.energies[15] = vec![
        energy![2, 1, 3],
        energy![2, 1, 3],

        //energy![1], energy![3], energy![0, 0, 3], energy![],
        //energy![1], energy![3], energy![0, 0, 3], energy![],
    ];
        */
    //println!("{:#?}", game);
    let mut runner = game.get_gpu_runner().await;
    runner.execute_gpu().await.unwrap();
}

fn main() {
    pollster::block_on(execute_reachability_small());
}
