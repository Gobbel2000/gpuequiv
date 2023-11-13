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
            (2, 3, update![0, Upd::Min(1)]),
            (3, 4, update![0, 0, 0, -1]),
            (4, 5, update![-1, 0]),
            (5, 6, update![0, -1]),
            (6, 7, update![-1, 0]),
            (6, 8, update![0, 0, Upd::Min(1)]),
            (8, 4, update![0, -1, 0]),
            (1, 9, update![0, 0, -1]),
            (9, 10, update![0, 0, -1]),
            (10, 11, update![0, 0, 0, -1]),
            (11, 3, update![Upd::Min(3)]),
            (10, 12, update![0, 0, -1]),
            (12, 10, update![0, 0, -1]),
            (0, 13, update![Upd::Min(2)]),
            (13, 14, update![-1]),
            (14, 15, update![0, -1]),
            (14, 16, update![0, 0, -1]),
            (16, 17, update![0, 0, -1]),
        ],
        &attacker_pos,
    );
    EnergyGame::standard_reach(graph)
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
    let mut game = EnergyGame::standard_reach(graph);
    assert_eq!(game.run().await.unwrap(),
        &[
            vec![energy![1, 0, 0, 1], energy![0, 1, 0, 1]],
            vec![energy![1], energy![0, 1]],
            vec![energy![0]],
            vec![energy![0]],
        ]
    );
}


fn combinations_graph() -> EnergyGame {
    let attacker_pos: Vec<bool> = (0..20)
        .map(|i| [1, 2, 3, 4, 6, 10, 16].contains(&i))
        .collect();
    let graph = GameGraph::new(
        20,
        &[
            (0, 1, update![-1]),
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
        &attacker_pos,
    );
    EnergyGame::standard_reach(graph)
}

#[pollster::test]
async fn combinations() {
    let mut game = combinations_graph();
    assert_eq!(game.run().await.unwrap(),
        &[
            vec![energy![3, 1, 1, 1, 0, 2], energy![2, 1, 2, 1, 0, 2]],
            vec![energy![2, 1, 1], energy![3, 0, 1], energy![1, 0, 2]],
            vec![energy![0, 0, 1], energy![0, 1]],
            vec![energy![0, 0, 0, 2, 2], energy![0, 0, 1], energy![0, 1]],
            vec![energy![0, 0, 0, 0, 0, 1]],
            vec![energy![1, 1, 1], energy![2, 0, 1]],
            vec![energy![0, 1], energy![1]],
            vec![Energy::zero()],
            vec![Energy::zero()],
            vec![energy![1, 0, 1]],
            vec![energy![1]],
            vec![Energy::zero()],
            vec![Energy::zero()],
            vec![Energy::zero()],
            vec![Energy::zero()],
            vec![energy![0, 0, 0, 2]],
            vec![energy![0, 0, 0, 1]],
            vec![Energy::zero()],
            vec![Energy::zero()],
            vec![Energy::zero()],
        ]
    );
}


// Create a very regular graph with adjustable size to test correctness of larger inputs.
//
// `single` is the number of 2-deep paths that extend from the root node.
// `double` is the number of subtrees under the root node that split into two leaf nodes at depth
// 2. Each of these double subtrees create 2 incomparable energy tuples, causing the root node
// (which is a defense node) to double the necessary number of combinations. Thus, `double` should
// be set very carefully, because the root node will have to perform 2**double combinations.
fn wide(single: u32, double: u32) -> (EnergyGame, Vec<Vec<Energy>>) {
    let n_nodes = 2 * single + 3 * double + 1;
    let mut edges = Vec::with_capacity(n_nodes as usize - 1);
    let mut attacker_pos = vec![false; n_nodes as usize];
    let mut expected = vec![vec![Energy::zero()]; n_nodes as usize];
    expected[0] = vec![energy![1, 1, 2], energy![1, 1, 1, 1]];
    for i in 1..=single {
        edges.push((0, i, update![Upd::Min(2)]));
        edges.push((i, i + single, update![-1]));
        attacker_pos[i as usize] = true;
        expected[i as usize] = vec![energy![1]];
    }
    for i in 1..=double {
        edges.push((0, 2*single + i, update![0, 0, -1]));
        edges.push((2*single + i, 2*single + double + i, update![0, 0, -1]));
        edges.push((2*single + i, 2*single + double + i + 1, update![0, 0, 0, -1]));
        attacker_pos[(2*single + i) as usize] = true;
        expected[(2*single + i) as usize] = vec![energy![0, 0, 1], energy![0, 0, 0, 1]];
    }
    let graph = GameGraph::new(
        n_nodes,
        &edges,
        &attacker_pos,
    );

    (EnergyGame::standard_reach(graph), expected)
}

#[pollster::test]
async fn wide_game() {
    let (mut game, expected) = wide(4100, 14);
    assert_eq!(game.run().await.unwrap(), expected);
}
