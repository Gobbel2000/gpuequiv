// Using the same input LTS as in the `build` example,
// run the full algorithm and use the output to compare two processes.

use gpuequiv::*;

async fn run() -> Result<()> {
    let lts = TransitionSystem::new(
        3,
        vec![
            (0, 0, 0),
            (0, 2, 0),
            (0, 2, 1),
            (1, 1, 0),
            (1, 2, 1),
            (2, 2, 0),
        ],
    );
    // Compare processes 0 and 1 (named S and S')
    let winning_budgets = lts.winning_budgets(0, 1).await?;
    for e in &winning_budgets {
        println!("{e}");
    }
    // The comparison for process S and S' is located at index 0.
    // `energy` now represents the minimal costs of a formula that can distinguish S and S'.
    let energy = &winning_budgets[0];
    println!("\nWinning budgets when comparing S and S':\n{}", energy);

    // Test for specific equivalences
    println!("Trace Equivalence between S and S': {}",
             energy.test_equivalence(std_equivalences::traces()));
    println!("S can simulate S': {}",
             energy.test_equivalence(std_equivalences::simulation()));
    println!("Failure Equivalence between S and S': {}",
             energy.test_equivalence(std_equivalences::failures()));
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    pollster::block_on(run())
}
