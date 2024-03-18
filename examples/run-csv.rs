use std::ffi::OsStr;
use std::env;
use std::io;
use std::time::Instant;

use gpuequiv::*;
use gpuequiv::energygame::*;
use gpuequiv::gamebuild::*;

async fn csv_lts(fname: &OsStr) -> io::Result<()> {
    let full_lts = TransitionSystem::from_csv_file(fname)?;
    let (lts, _bisim) = full_lts.bisimilar_minimize();

    let now = Instant::now();
    let (builder, start_info) = GameBuild::compare_all(&lts);
    println!("Game built in {:.5}s", now.elapsed().as_secs_f64());

    println!("Number of starting positions: {}", start_info.starting_points.len());
    println!("Number of nodes: {}", builder.game.n_vertices());
    println!("Number of edges: {}", builder.game.column_indices.len());

    // Create EnergyGame from built game graph
    let energy_game = EnergyGame::standard_reach(builder.game);

    // Solve energy game on GPU
    println!("Running game...");
    let mut runner = energy_game.get_gpu_runner().await.unwrap();
    let now = Instant::now();
    runner.execute_gpu().await.unwrap();
    println!("Ran energy game in {:.5}s", now.elapsed().as_secs_f64());

    let equivalence = start_info.equivalence(runner.game.energies);

    // Generate various equivalence relations
    println!("\nNumber of equivalence classes according to different equivalences:");
    let enabledness = equivalence.relation(std_equivalences::enabledness());
    let traces = equivalence.relation(std_equivalences::traces());
    let simulation = equivalence.relation(std_equivalences::simulation());
    println!("Enabledness:  {}", enabledness.count_classes());
    println!("Traces:       {}", traces.count_classes());
    println!("Simulation:   {}", simulation.count_classes());
    println!("Bisimulation: {}", lts.n_vertices());
    Ok(())
}

fn main() -> io::Result<()> {
    env_logger::init();
    if let Some(fname) = env::args_os().nth(1) {
        pollster::block_on(csv_lts(&fname))
    } else {
        eprintln!("Invalid arguments. Usage: {} [CSV-FILE]",
                  env::args().next().unwrap_or_default());
        std::process::exit(2);
    }
}
