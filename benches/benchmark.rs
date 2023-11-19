use std::fs::File;

use gpuequiv::*;

use criterion::{criterion_group, criterion_main, Criterion, async_executor::AsyncExecutor};

pub struct PollsterExecutor;
impl AsyncExecutor for PollsterExecutor {
    fn block_on<T>(&self, future: impl std::future::Future<Output = T>) -> T {
        pollster::block_on(future)
    }
}

pub fn benchmark(c: &mut Criterion) {
    let reader = File::open("benches/g10000.json").unwrap();
    let graph: GameGraph = serde_json::from_reader(&reader).unwrap();
    c.bench_function("10k_nodes", |b| b.to_async(PollsterExecutor).iter(|| bench_inner(graph.clone())));
}

async fn bench_inner(graph: GameGraph) {
    let mut game = EnergyGame::standard_reach(graph);
    game.run().await.unwrap();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
