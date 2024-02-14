// This example shows how to compare all processes of
// an LTS with each other and interprete the result.
//
// This example is also set up to run on wasm.

use gpuequiv::*;

// Redirect println to browser log on wasm
#[cfg(target_arch = "wasm32")]
macro_rules! println {
    ($($arg:tt)+) => (log::info!($($arg)+))
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
fn lts() -> TransitionSystem {
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

async fn run() -> Result<()> {
    let lts = lts();

    // Compare all processes with each other
    let (equivalence, minimization) = lts.equivalences().await?;

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

    let failures = equivalence.relation(std_equivalences::failures())
        .with_mapping(&minimization); // Apply mapping to have an equivalence relation to the original,
                                      // unminimized LTS.
    // Construct the equivalence classes themselves
    println!("\nFailure equivalence classes:");
    for class in failures.get_classes() {
        println!("{class:?}");
    }

    // Compare two specific processes using the equivalence relation,
    // minimization was already applied on the whole relation.
    println!("\nFailure Equivalence between processes 0 and 9: {}",
             failures.equiv(0, 9));

    // Directly compare 2 processes. The minimization mapping is applied manually.
    println!("Failure Trace Equivalence between processes 0 and 9: {}",
             equivalence.equiv(minimization[0] as u32, minimization[9] as u32,
                               std_equivalences::failure_traces()));
    // Preorder comparison
    println!("Process 0 can simulate process 9: {}",
             equivalence.preorder(minimization[0] as u32, minimization[9] as u32,
                                  std_equivalences::simulation()));
    Ok(())
}

fn main() -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run())
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("Could not initialize console logger");
        async fn run_panicking() {
            run().await.unwrap()
        }
        wasm_bindgen_futures::spawn_local(run_panicking());
        Ok(())
    }
}
