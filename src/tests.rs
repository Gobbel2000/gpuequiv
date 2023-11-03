use super::*;

fn simple_graph() -> EnergyGame {
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

#[pollster::test]
async fn simple() {
    let mut game = simple_graph();
    assert_eq!(game.run().await.unwrap(),
        &[
           vec![energy![1, 1]], // 0
           vec![], // 1
           vec![energy![2, 1, 0, 1]], // 2
           vec![energy![2, 1, 0, 1]], // 3
           vec![energy![2, 1]], // 4
           vec![energy![1, 1]], // 5
           vec![energy![1]], // 6
           vec![energy![0]], // 7
           vec![energy![2, 2]], // 8
           vec![], // 9
           vec![], // 10
           vec![energy![2, 1, 2, 1]], // 11
           vec![], // 12
           vec![energy![1, 1]], // 13
           vec![energy![0, 1]], // 14
           vec![energy![0]], // 15
           vec![], // 16
           vec![], // 17
        ]
    );
}

#[pollster::test]
async fn simple_one_dimension() {
    let mut game = simple_graph();
    // Set all updates (edge weights) to -1, 0, 0, ...
    for node in game.graph.weights.iter_mut() {
        for w in node.iter_mut() {
            *w = update![-1];
        }
    }
    // Energies saturate at 3
    assert_eq!(game.run().await.unwrap(),
        &[
            vec![energy![3]],
            vec![],
            vec![energy![3]],
            vec![energy![3]],
            vec![energy![3]],
            vec![energy![2]],
            vec![energy![1]],
            vec![energy![0]],
            vec![energy![3]],
            vec![],
            vec![],
            vec![energy![3]],
            vec![],
            vec![energy![2]],
            vec![energy![1]],
            vec![energy![0]],
            vec![],
            vec![],
        ]
    );
}

// Test a small example where a multi-energy pareto front is created
#[pollster::test]
async fn fork() {
    let graph = GameGraph::new(
        4,
        &[
            (0, 1, update![0, 0, 0, -1]),
            (1, 2, update![-1, 0]),
            (1, 3, update![0, -1]),
        ],
        &[false, true, false, false],
    );
    let mut game = EnergyGame::from_graph(graph)
        .with_reach(vec![2, 3]);
    assert_eq!(game.run().await.unwrap(),
        &[
            vec![energy![1, 0, 0, 1], energy![0, 1, 0, 1]],
            vec![energy![1], energy![0, 1]],
            vec![energy![0]],
            vec![energy![0]],
        ]
    );
}
