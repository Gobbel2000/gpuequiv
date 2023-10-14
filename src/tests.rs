use super::*;

#[test]
fn reachability_small() {
    pollster::block_on(execute_reachability_small());
}

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
            (6, 7, 0.into()),
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
    let mut game = EnergyGame::from_graph(example_graph());
    game.to_reach = vec![7, 15];
    let mut runner = game.get_gpu_runner().await;
    assert_eq!(runner.execute_gpu().await, Ok(vec![
        Some(3),
        None,
        Some(5),
        Some(4),
        Some(3),
        Some(2),
        Some(1),
        Some(0),
        Some(4),
        None,
        None,
        Some(5),
        None,
        Some(2),
        Some(1),
        Some(0),
        None,
        None,
    ]));
}
