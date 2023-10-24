use gpuequiv::*;

// Only one-dimensional energies
fn example_game() -> EnergyGame {
    //let attacker_pos = vec![true; 18];
    let attacker_pos: Vec<bool> = (0..18)
        .map(|i| [0, 2, 4, 6, 9, 11, 12, 14, 17].contains(&i))
        .collect();
    let graph = GameGraph::new(
        18,
        &[
            (0, 1, update![-1]),
            (1, 2, update![-1]),
            (2, 3, update![-1]),
            (3, 4, update![-1]),
            (4, 5, update![-1]),
            (5, 6, update![-1]),
            (6, 7, update![-1]),
            (6, 8, update![-1]),
            (8, 4, update![-1]),
            (1, 9, update![-1]),
            (9, 10, update![-1]),
            (10, 11, update![-1]),
            (11, 3, update![-1]),
            (10, 12, update![-1]),
            (12, 10, update![-1]),
            (0, 13, update![-1]),
            (13, 14, update![-1]),
            (14, 15, update![-1]),
            (14, 16, update![-1]),
            (16, 17, update![-1]),
        ],
        &attacker_pos,
    );
    EnergyGame::from_graph(graph)
        .with_reach(vec![7, 15])
}

// Varied, multidimensional updates, leading to multidimensional energies
fn multidimensional() -> EnergyGame {
    let attacker_pos: Vec<bool> = (0..18)
        .map(|i| [0, 2, 4, 6, 9, 11, 12, 14, 17].contains(&i))
        .collect();
    let graph = GameGraph::new(
        18,
        &[
            (0, 1, update![0, 0, -1]),
            (1, 2, update![0, -1, 0]),
            (2, 3, update![0, Upd::Min(0)]),
            (3, 4, update![0, 0, 0, -1]),
            (4, 5, update![-1, 0]),
            (5, 6, update![0, -1]),
            (6, 7, update![-1, 0]),
            (6, 8, update![0, 0, Upd::Min(0)]),
            (8, 4, update![0, -1, 0]),
            (1, 9, update![0, 0, -1]),
            (9, 10, update![0, 0, -1]),
            (10, 11, update![0, 0, 0, -1]),
            (11, 3, update![Upd::Min(2)]),
            (10, 12, update![0, 0, -1]),
            (12, 10, update![0, 0, -1]),
            (0, 13, update![Upd::Min(1)]),
            (13, 14, update![-1]),
            (14, 15, update![0, -1]),
            (14, 16, update![0, 0, -1]),
            (16, 17, update![0, 0, -1]),
        ],
        &attacker_pos,
    );
    EnergyGame::from_graph(graph)
        .with_reach(vec![7, 15])
}

// Pareto front with multiple energy tuples
fn multifront() -> EnergyGame {
    let graph = GameGraph::new(
        4,
        &[
            (0, 1, update![0, 0, 0, -1]),
            (1, 2, update![-1, 0]),
            (1, 3, update![0, -1]),
        ],
        &[false, true, false, false],
    );
    EnergyGame::from_graph(graph)
        .with_reach(vec![2, 3])
}

async fn execute() {
    let mut game = multifront();
    let mut runner = game.get_gpu_runner().await;
    runner.execute_gpu().await.unwrap();
    println!("{:#?}", game.energies);
}

fn main() {
    pollster::block_on(execute());
}
