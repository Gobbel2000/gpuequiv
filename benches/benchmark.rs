use gpuequiv::*;
use gpuequiv::gamebuild::*;
use gpuequiv::energygame::*;

use criterion::{criterion_group, criterion_main, async_executor::AsyncExecutor,
                Criterion, BatchSize, BenchmarkId, PlotConfiguration, AxisScale};

struct PollsterExecutor;
impl AsyncExecutor for PollsterExecutor {
    fn block_on<T>(&self, future: impl std::future::Future<Output = T>) -> T {
        pollster::block_on(future)
    }
}

fn vlts(c: &mut Criterion) {
    static INPUTS: [&str; 10] = [
        "vasy_25_25.csv",
        "vasy_0_1.csv",
        "vasy_1_4.csv",
        "vasy_5_9.csv",
        "peterson_mutex_weak.csv",
        "vasy_8_38.csv",
        "cwi_3_14.csv",
        "vasy_8_24.csv",
        "vasy_10_56.csv",
        "cwi_1_2.csv",
        //TOO LARGE: "vasy_18_73.csv",
    ];

    let mut group = c.benchmark_group("vlts");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for name in INPUTS {
        let full_lts = TransitionSystem::from_csv_file("benches/lts/".to_string() + name).unwrap();
        let (lts, _bisim) = full_lts.bisimilar_minimize();

        let (builder, _start_info) = GameBuild::compare_all(&lts);
        let game_graph = builder.game;
        let n_edges = game_graph.column_indices.len();
        println!("\n{name}\n====================");
        println!("Number of nodes: {}", game_graph.n_vertices());
        println!("Number of edges: {}", n_edges);
        let energy_game = EnergyGame::standard_reach(game_graph);

        // Adjust sample count to keep runtime roughly below 1min for each measurement.
        // 60 Million is an estimate for the number of edges we can process in a minute.
        // Criterion requires a minimum of 10 samples.
        group.sample_size((60_000_000 / n_edges.max(1)).clamp(10, 100));

        // Measure only energy game
        group.bench_with_input(BenchmarkId::new("energy game", n_edges),
            &energy_game,
            |b, energy_game| b.to_async(PollsterExecutor)
                .iter_batched(|| energy_game.clone(),
                    |energy_game| async move {
                        energy_game.run().await.unwrap();
                    },
                    BatchSize::LargeInput,
                ),
            );
        drop(energy_game);

        // Complete algorithm, including game build
        group.bench_with_input(BenchmarkId::new("complete", n_edges),
            &lts,
            |b, lts| b.to_async(PollsterExecutor)
                .iter_with_large_drop(|| lts.equivalences()),
        );

    }
    group.finish();
}

criterion_group!(benches, vlts);
criterion_main!(benches);
