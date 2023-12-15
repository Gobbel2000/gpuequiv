use super::*;

fn simple_graph() -> EnergyGame {
    let conf = EnergyConf::STANDARD;
    let attacker_pos: Vec<bool> = (0..18)
        .map(|i| [0, 2, 4, 6, 9, 11, 12, 14, 17].contains(&i))
        .collect();
    let graph = GameGraph::new(
        18,
        vec![
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
        attacker_pos,
        conf,
    );
    EnergyGame::standard_reach(graph)
}

macro_rules! earray {
    ($conf:expr, $( $x:expr ),* ) => {
        EnergyArray::from_conf([ $( $x, )* ].as_slice(), $conf).unwrap()
    }
}

#[pollster::test]
async fn simple() {
    let c = EnergyConf::STANDARD;
    let mut game = simple_graph();
    assert_eq!(game.run().await.unwrap(),
        &[
            earray![c, vec![1, 1]],
            EnergyArray::empty(c),
            earray![c, vec![2, 1, 0, 1]],
            earray![c, vec![2, 1, 0, 1]],
            earray![c, vec![2, 1]],
            earray![c, vec![1, 1]],
            earray![c, vec![1]],
            EnergyArray::zero(1, c),
            earray![c, vec![2, 2]],
            EnergyArray::empty(c),
            EnergyArray::empty(c),
            earray![c, vec![2, 1, 2, 1]],
            EnergyArray::empty(c),
            earray![c, vec![1, 1]],
            earray![c, vec![0, 1]],
            EnergyArray::zero(1, c),
            EnergyArray::empty(c),
            EnergyArray::empty(c),
        ]
    );
}

#[pollster::test]
async fn simple_one_dimension() {
    let c = EnergyConf::STANDARD;
    let mut game = simple_graph();
    // Set all updates (edge weights) to -1, 0, 0, ...
    for array in game.graph.weights.iter_mut() {
        for i in 0..array.n_updates() {
            array.set(i, Update::new(&[Upd::Decrement], array.get_conf()).unwrap());
        }
    }
    // Energies saturate at 3
    assert_eq!(game.run().await.unwrap(),
        &[
            earray![c, vec![3]],
            EnergyArray::empty(c), 
            earray![c, vec![3]],
            earray![c, vec![3]],
            earray![c, vec![3]],
            earray![c, vec![2]],
            earray![c, vec![1]],
            earray![c, vec![0]],
            earray![c, vec![3]],
            EnergyArray::empty(c), 
            EnergyArray::empty(c), 
            earray![c, vec![3]],
            EnergyArray::empty(c), 
            earray![c, vec![2]],
            earray![c, vec![1]],
            earray![c, vec![0]],
            EnergyArray::empty(c), 
            EnergyArray::empty(c), 
        ]
    );
}

// Test a small example where a multi-energy pareto front is created
#[pollster::test]
async fn fork() {
    let c = EnergyConf::STANDARD;
    let graph = GameGraph::new(
        4,
        vec![
            (0, 1, update![0, 0, 0, -1]),
            (1, 2, update![-1, 0]),
            (1, 3, update![0, -1]),
        ],
        [false, true, false, false].to_vec(),
        c,
    );
    let mut game = EnergyGame::standard_reach(graph);
    assert_eq!(game.run().await.unwrap(),
        &[
            earray![c, vec![1, 0, 0, 1], vec![0, 1, 0, 1]],
            earray![c, vec![1], vec![0, 1]],
            earray![c, vec![0]],
            earray![c, vec![0]],
        ]
    );
}


fn combinations_graph() -> EnergyGame {
    let conf = EnergyConf::STANDARD;
    let attacker_pos: Vec<bool> = (0..20)
        .map(|i| [1, 2, 3, 4, 6, 10, 16].contains(&i))
        .collect();
    let graph = GameGraph::new(
        20,
        vec![
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
        attacker_pos,
        conf,
    );
    EnergyGame::standard_reach(graph)
}

#[pollster::test]
async fn combinations() {
    let c = EnergyConf::STANDARD;
    let mut game = combinations_graph();
    assert_eq!(game.run().await.unwrap(),
        &[
            earray![c, vec![3, 1, 1, 1, 0, 2], vec![2, 1, 2, 1, 0, 2]],
            earray![c, vec![2, 1, 1], vec![3, 0, 1], vec![1, 0, 2]],
            earray![c, vec![0, 0, 1], vec![0, 1]],
            earray![c, vec![0, 0, 0, 2, 2], vec![0, 0, 1], vec![0, 1]],
            earray![c, vec![0, 0, 0, 0, 0, 1]],
            earray![c, vec![1, 1, 1], vec![2, 0, 1]],
            earray![c, vec![0, 1], vec![1]],
            earray![c, vec![0]],
            earray![c, vec![0]],
            earray![c, vec![1, 0, 1]],
            earray![c, vec![1]],
            earray![c, vec![0]],
            earray![c, vec![0]],
            earray![c, vec![0]],
            earray![c, vec![0]],
            earray![c, vec![0, 0, 0, 2]],
            earray![c, vec![0, 0, 0, 1]],
            earray![c, vec![0]],
            earray![c, vec![0]],
            earray![c, vec![0]],
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
fn wide(single: u32, double: u32) -> (EnergyGame, Vec<EnergyArray>) {
    let c = EnergyConf::STANDARD;
    let n_nodes = 2 * single + 3 * double + 1;
    let mut edges = Vec::with_capacity(n_nodes as usize - 1);
    let mut attacker_pos = vec![false; n_nodes as usize];
    let mut expected = vec![EnergyArray::zero(1, c); n_nodes as usize];
    expected[0] = earray![c, vec![1, 1, 2], vec![1, 1, 1, 1]];
    for i in 1..=single {
        edges.push((0, i, update![Upd::Min(2)]));
        edges.push((i, i + single, update![-1]));
        attacker_pos[i as usize] = true;
        expected[i as usize] = earray![c, vec![1]];
    }
    for i in 1..=double {
        edges.push((0, 2*single + i, update![0, 0, -1]));
        edges.push((2*single + i, 2*single + double + i, update![0, 0, -1]));
        edges.push((2*single + i, 2*single + double + i + 1, update![0, 0, 0, -1]));
        attacker_pos[(2*single + i) as usize] = true;
        expected[(2*single + i) as usize] = earray![c, vec![0, 0, 1], vec![0, 0, 0, 1]];
    }
    let graph = GameGraph::new(
        n_nodes,
        edges,
        attacker_pos,
        c,
    );

    (EnergyGame::standard_reach(graph), expected)
}

#[pollster::test]
async fn wide_game() {
    let (mut game, expected) = wide(4100, 14);
    assert_eq!(game.run().await.unwrap(), expected);
}
