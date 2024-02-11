// Using the same input LTS as in the `build` example,
// run the full algorithm and use the output to compare two processes.

use gpuequiv::*;

async fn run() -> Result<()> {
    let lts = TransitionSystem::from(
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
    let winning_budgets = lts.compare(0, 1).await?;
    // `winning_budgets` now represents the minimal costs of a formula that can distinguish S and S'.
    println!("\nWinning budgets when comparing S and S':\n{winning_budgets}", );

    // Test for specific equivalences
    println!("Trace Equivalence between S and S': {}",
             winning_budgets.test_equivalence(std_equivalences::traces()));
    println!("S can simulate S': {}",
             winning_budgets.test_equivalence(std_equivalences::simulation()));
    println!("Failure Equivalence between S and S': {}",
             winning_budgets.test_equivalence(std_equivalences::failures()));
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    pollster::block_on(run())
}
