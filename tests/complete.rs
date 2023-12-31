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
            &Position::defend(0, vec![], vec![1]),  // (S, {}, {S'}) d
            &Position::defend(2, vec![1], vec![]),
            &Position::defend(2, vec![], vec![1]),
            &Position::defend(2, vec![2], vec![]),
            &Position::defend(2, vec![], vec![2]),
            &Position::clause(0, 1),  // (S, S') a^
            &Position::clause(2, 1),  // (Div, S') a^
            &Position::clause(2, 2),
            &Position::attack(1, vec![0]),
            &Position::attack(1, vec![2]),
            &Position::attack(1, vec![0, 2]),  // (S', {S, Div}) a
            &Position::defend(1, vec![0], vec![]),
            &Position::defend(1, vec![], vec![0]),
            &Position::attack(2, vec![]),
            &Position::defend(1, vec![2], vec![]),
            &Position::defend(1, vec![], vec![2]),
            &Position::defend(1, vec![0, 2], vec![]),
            &Position::defend(1, vec![], vec![0, 2]),
            &Position::defend(1, vec![2], vec![0]),
            &Position::clause(1, 0),
            &Position::defend(2, vec![], vec![]),
            &Position::clause(1, 2),
        ],
    );
    let adj_expected = vec![
        vec![0, 1, 2, 3, 4],
        vec![1, 5, 6],
        vec![2, 7, 8],
        vec![9],
        vec![0],
        vec![10],
        vec![1],
        vec![11],
        vec![2],
        vec![0, 12],
        vec![1, 13],
        vec![2],
        vec![2, 14, 15, 16],
        vec![13, 17, 18, 19],
        vec![2, 14, 20, 21, 22],
        vec![23],
        vec![12],
        vec![17, 24],
        vec![25],
        vec![13],
        vec![23, 25],
        vec![14],
        vec![12, 25],
        vec![0, 12],
        vec![],
        vec![1, 13],
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
            earray!(c, vec![2, 2, 2, 0, 1, 1]),
            earray!(c, vec![1, 1, 0, 0, 1, 1]),
            earray!(c, vec![1, 2, 1, 0, 1, 1]),
            EnergyArray::empty(c),
            EnergyArray::empty(c),
            earray!(c, vec![2, 2, 0, 2, 1, 1], vec![2, 3, 0, 0, 2, 3]),
            earray!(c, vec![1, 1, 0, 0, 1, 1]),
            EnergyArray::empty(c),
            earray!(c, vec![2, 3, 0, 0, 2, 2]),
            earray!(c, vec![1, 1]),
            earray!(c, vec![2, 3, 0, 0, 2, 2]),
            earray!(c, vec![2, 2, 0, 0, 2, 2]),
            earray!(c, vec![2, 3, 2, 0, 2, 2]),
            earray!(c, vec![0, 1]),
            earray!(c, vec![1, 1, 1, 1], vec![1, 2, 0, 0, 1, 2]),
            earray!(c, vec![1, 1, 1]),
            earray!(c, vec![2, 2, 0, 0, 2, 2]),
            earray!(c, vec![2, 3, 2, 0, 2, 2]),
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
