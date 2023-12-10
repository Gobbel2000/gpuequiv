use gpuequiv::challenge_wgpu::*;
use std::time::Instant;

async fn run() {
    let lts = TransitionSystem::new(
        11,
        vec![
            (0, 1, 1),
            (0, 2, 1),
            (0, 2, 2),
            (0, 1, 3),
            (0, 2, 3),
            (0, 2, 4),
            (0, 1, 5),
            (0, 2, 5),

            (1, 3, 1),
            (1, 3, 2),
            (1, 3, 3),

            (2, 3, 1),
            (2, 3, 2),
            (2, 3, 3),

            (3, 1, 2),

            (4, 0, 4),
            (5, 0, 4),
            (6, 0, 4),

            (7, 1, 1),
            (8, 1, 1),
            (9, 1, 1),

            (10, 3, 1),
            (10, 3, 2),
            (10, 3, 3),
            (10, 3, 4),
        ],
    );
    let runner = GameBuild::with_lts(lts).await.unwrap();
    let now = Instant::now();
    for _ in 0..1000 {
        runner.execute_gpu().await.unwrap();
    }
    let elapsed = now.elapsed();
    println!("Took {}ms", elapsed.as_millis());
}

fn main() {
    pollster::block_on(run())
}
