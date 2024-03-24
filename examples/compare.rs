// Run the full algorithm and use the output to compare two processes.

use gpuequiv::{TransitionSystem, Result, std_equivalences};

async fn run() -> Result<()> {
    // This LTS corresponds to Figure 2.1 in the thesis text at
    // https://github.com/Gobbel2000/thesis-gpuequiv/releases/latest/download/thesis.pdf
    let (a, b, c) = (0, 1, 2);
    let lts = TransitionSystem::from(vec![
     // (Start, End, Action)
        (0, 1, a),
        (1, 2, b),
        (1, 3, c),

        (4, 5, a),
        (5, 7, b),
        (4, 6, a),
        (6, 8, c),
    ]);

    // Run the algorithm on the GPU
    let energies = lts.compare(0, 4).await?;
    println!("{energies}");

    // Test various equivalences between processes 0 and 4
    println!("Trace: {}",
        energies.test_equivalence(std_equivalences::traces()));
    println!("Simulation: {}",
        energies.test_equivalence(std_equivalences::simulation()));
    println!("Bisimulation: {}",
        energies.test_equivalence(std_equivalences::bisimulation()));

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    pollster::block_on(run())
}
