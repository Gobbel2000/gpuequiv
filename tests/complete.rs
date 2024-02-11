use gpuequiv::*;
use gpuequiv::energygame::EnergyGame;
use gpuequiv::gamebuild::*;

// Example transition system from
// Bisping 2023 - Process Equivalence Problems as Energy Games,
// Figure 4
// The expected game graph and energies correspond to Figure 8.
fn example() -> TransitionSystem {
    TransitionSystem::from(
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
    let builder = GameBuild::compare(&lts, 0, 1);
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
    assert_eq!(builder.game.n_vertices() as usize, adj_expected.len());
    for (i, exp) in adj_expected.into_iter().enumerate() {
        // Compare adjacency lists without regard to ordering
        let mut adj_copy = builder.game.adj(i as u32).to_vec();
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
    let builder = GameBuild::compare(&lts, 0, 1);
    let mut game = EnergyGame::standard_reach(builder.game);
    game.run().await.unwrap();
    for (i, (en, exp)) in game.energies.into_iter().zip(expected).enumerate() {
        assert_eq!(en, exp, "Wrong energies for node {i}: {en} != {exp}");
    }
}

// Exercise 3.5 (page 48) from L. Aceto et. al. - Reactive Systems: Modelling, Specification and Verification
// s0 and t0 (here 0 and 5) are bisimilar.
fn bisimilar() -> TransitionSystem {
    TransitionSystem::from(
        // Action a => 0
        //        b => 1
        vec![
            (0, 1, 0), // proc 0..=4 is s0..=s4
            (0, 2, 0),
            (1, 3, 0),
            (1, 4, 1),
            (2, 4, 0),
            (3, 0, 0),
            (4, 0, 0),

            (5, 6, 0), // proc 5..=9 is t0..=t4
            (5, 8, 0),
            (6, 7, 0),
            (6, 7, 1),
            (7, 5, 0),
            (8, 9, 0),
            (9, 0, 0),
        ],
    )
}

#[pollster::test]
async fn test_bisimilar() {
    let lts = bisimilar();
    let (partitions, _count) = lts.signature_refinement();
    assert!(partitions[0] == partitions[5]);
    let energies = lts.compare(0, 5).await.unwrap();
    assert_eq!(energies, EnergyArray::empty(EnergyConf::STANDARD));
    assert!(energies.test_equivalence(std_equivalences::bisimulation()));
}

// Counterexample 3 (page 286) from R.J.v. Glabbeek - The Linear Time - Branching Time Spectrum
// 0 ==Failure 9
// 0 !=Failure Traces 9
// 0 ==Readiness 9
// 0 !=Readiness Traces 9
//
// 0 := a(b + cd) + a(f + ce)
// 9 := a(b + ce) + a(f + cd)
//
// a..=f => 0..=5
fn failure_trace() -> TransitionSystem {
    TransitionSystem::from(
        vec![
            (0, 1, 0),
            (0, 5, 0),
            (1, 2, 1),
            (1, 3, 2),
            (3, 4, 3),

            (5, 6, 5),
            (5, 7, 2),
            (7, 8, 4),


            (9, 10, 0),
            (9, 14, 0),
            (10, 11, 1),
            (10, 12, 2),
            (12, 13, 4), // Here e (4) instead of d (3)

            (14, 15, 5),
            (14, 16, 2),
            (16, 17, 3),
        ],
    )
}

#[pollster::test]
async fn test_failure_trace() {
    let lts = failure_trace();
    let budgets = &lts.compare(0, 9).await.unwrap();
    // Equivalences that hold
    assert!(budgets.test_equivalence(std_equivalences::failures()));
    assert!(budgets.test_equivalence(std_equivalences::readiness()));
    assert!(budgets.test_equivalence(std_equivalences::revivals()));
    assert!(budgets.test_equivalence(std_equivalences::traces()));
    // Equivalences that don't hold
    assert!(!budgets.test_equivalence(std_equivalences::failure_traces()));
    assert!(!budgets.test_equivalence(std_equivalences::readiness_traces()));
    assert!(!budgets.test_equivalence(std_equivalences::ready_simulation()));
    assert!(!budgets.test_equivalence(std_equivalences::bisimulation()));
}
