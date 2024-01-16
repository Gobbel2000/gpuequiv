use gpuequiv::*;
use gpuequiv::gamebuild::*;
use gpuequiv::gamebuild::gamepos::*;

// Example transition system from
// Bisping 2023 - Process Equivalence Problems as Energy Games,
// Figure 4
// The expected game graph and energies correspond to Figure 8.
fn example() -> TransitionSystem {
    TransitionSystem::new(
        3,
        vec![
            (0, 0, 0),  // 0 => S
            (0, 2, 0),
            (0, 2, 1),
            (1, 1, 0),  // 1 => S'
            (1, 2, 1),
            (2, 2, 0),  // 2 => Div
            // label 0 => τ
            //       1 => ecA
        ],
    )
}

#[test]
fn example_game_graph() {
    let lts = example();
    let mut builder = GameBuild::with_lts(lts);
    builder.build(0, 1);
    assert_eq!(builder.nodes.iter().map(|p| p.as_ref()).collect::<Vec<_>>(),
        vec![
            &Position::attack(0, vec![1]), // (S, {S'}) a
            &Position::attack(2, vec![1]), // (Div, {S'}) a
            &Position::attack(2, vec![2]), // (Div, {Div}) a
            &Position::defend(0, vec![1], vec![]),  // (S, {S'}, {}) d
            &Position::defend(2, vec![1], vec![]),
            &Position::clause(0, 1),  // (S, S') a^
            &Position::clause(2, 1),  // (Div, S') a^
            &Position::attack(1, vec![0]),
            &Position::attack(1, vec![2]),
            &Position::attack(1, vec![0, 2]),  // (S', {S, Div}) a
            &Position::defend(1, vec![0], vec![]),
            &Position::attack(2, vec![]),
            &Position::defend(1, vec![2], vec![]),
            &Position::defend(1, vec![0, 2], vec![]),
            &Position::defend(1, vec![2], vec![0]),
            &Position::clause(1, 0),
            &Position::defend(2, vec![], vec![]),
            &Position::clause(1, 2),
        ],
    );
    let adj_expected = vec![
        vec![0, 1, 2, 3],
        vec![1, 4],
        vec![],
        vec![5],
        vec![6],
        vec![0, 7],
        vec![1, 8],
        vec![2, 9, 10],
        vec![8, 11, 12],
        vec![2, 9, 13, 14],
        vec![15],
        vec![11, 16],
        vec![17],
        vec![15, 17],
        vec![7, 17],
        vec![0, 7],
        vec![],
        vec![1, 8],
    ];
    assert_eq!(builder.game.adj.len(), adj_expected.len());
    for (i, (adj, exp)) in builder.game.adj.iter().zip(adj_expected).enumerate() {
        // Compare adjacency lists without regard to ordering
        let mut adj_copy = adj.clone();
        adj_copy.sort();
        assert_eq!(adj_copy, exp, "Adjacency list for node {i} is wrong");
    }
}

macro_rules! earray {
    ($conf:expr, $( $x:expr ),* ) => {
        EnergyArray::from_conf([ $( $x, )* ].as_slice(), $conf).unwrap()
    }
}

#[pollster::test]
async fn example_energies() {
    let lts = example();
    let c = EnergyConf::STANDARD;
    let expected = vec![
            earray!(c, vec![2, 2, 0, 0, 1, 1]),
            earray!(c, vec![1, 2, 0, 0, 1, 1]),
            EnergyArray::empty(c),  // (Div, {Div})a: Cannot be won by attacker
            earray!(c, vec![2, 2, 2, 2, 1, 1], vec![2, 3, 0, 0, 2, 3]),
            earray!(c, vec![1, 1, 0, 0, 1, 1]),
            earray!(c, vec![2, 2, 0, 2, 1, 1], vec![2, 3, 0, 0, 2, 3]),
            earray!(c, vec![1, 1, 0, 0, 1, 1]),
            earray!(c, vec![2, 3, 0, 0, 2, 2]),
            earray!(c, vec![1, 1]),
            earray!(c, vec![2, 3, 0, 0, 2, 2]),
            earray!(c, vec![2, 2, 0, 0, 2, 2]),
            earray!(c, vec![0, 1]),
            earray!(c, vec![1, 1, 1, 1], vec![1, 2, 0, 0, 1, 2]),
            earray!(c, vec![2, 2, 0, 0, 2, 2]),
            earray!(c, vec![2, 3, 2, 0, 2, 2]),
            earray!(c, vec![2, 2, 0, 0, 2, 2]),
            EnergyArray::zero(1, c),
            earray!(c, vec![1, 1, 0, 1], vec![1, 2, 0, 0, 1, 2]),
        ];
    for (i, (en, exp)) in lts.winning_budgets(0, 1).await.unwrap()
        .into_iter().zip(expected).enumerate()
    {
        assert_eq!(en, exp, "Wrong energies for node {i}: {en} != {exp}");
    }
}
