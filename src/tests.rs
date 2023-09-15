use super::*;

#[test]
fn reachability_small() {
    pollster::block_on(execute_reachability_small());
}

fn example_graph() -> GameGraph {
    let attacker_pos: Vec<bool> = (0..18)
        .map(|i| [0, 2, 4, 6, 9, 11, 12, 14, 17].contains(&i))
        .collect();
    GameGraph::new(
        18,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (6, 8),
            (8, 5),
            (1, 9),
            (9, 10),
            (10, 11),
            (11, 3),
            (10, 12),
            (12, 10),
            (0, 13),
            (13, 14),
            (14, 15),
            (14, 16),
            (16, 17),
        ],
        &attacker_pos,
    )
}

async fn execute_reachability_small() {
    let mut game = EnergyGame {
        graph: example_graph(),
    };
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
        Some(3),
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
