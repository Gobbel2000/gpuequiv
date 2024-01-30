// This example shows how to compare all processes of
// an LTS with each other and interprete the result.

use gpuequiv::*;

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
fn lts() -> TransitionSystem {
    TransitionSystem::new(
        18,
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

async fn run() -> Result<()> {
    let lts = lts();

    // Compare all processes with each other
    let equivalence = lts.equivalences().await?;

    // Look at the winning budgets of the game
    for (energy, position) in equivalence.energies.iter()
        .zip(&equivalence.start_info.starting_points)
    {
        println!("{position}");
        println!("{energy}\n");
    }

    // Build specific equivalence relations
    let bisimulation = equivalence.relation(std_equivalences::bisimulation());
    // Count the number of equivalence classes
    println!("Number of Bisimulation equivalence classes: {}", bisimulation.count_classes());

    let failures = equivalence.relation(std_equivalences::failures());
    // Construct the equivalence classes themselves
    println!("\nFailure equivalence classes:");
    for class in failures.get_classes() {
        println!("{class:?}");
    }

    // Compare two specific processes
    println!("\nFailure Equivalence between processes 0 and 9: {}",
             failures.equiv(0, 9));
    let failure_traces = equivalence.relation(std_equivalences::failure_traces());
    println!("Failure Trace Equivalence between processes 0 and 9: {}",
             failure_traces.equiv(0, 9));
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    pollster::block_on(run())
}
