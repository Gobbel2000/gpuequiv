use std::fs::File;
use std::env;
use std::io;

use gpuequiv::*;

// Varied, multidimensional updates, leading to multidimensional energies
fn _multidimensional() -> EnergyGame {
    let attacker_pos: Vec<bool> = (0..18)
        .map(|i| [0, 2, 4, 6, 9, 11, 12, 14, 17].contains(&i))
        .collect();
    let graph = GameGraph::new(
        18,
        &[
            (0, 1, update![0, 0, -1]),
            (1, 2, update![0, -1, 0]),
            (2, 3, update![0, Upd::Min(1)]),
            (3, 4, update![0, 0, 0, -1]),
            (4, 5, update![-1, 0]),
            (5, 6, update![0, -1]),
            (6, 7, update![-1, 0]),
            (6, 8, update![0, 0, Upd::Min(1)]),
            (8, 4, update![0, -1, 0]),
            (1, 9, update![0, 0, -1]),
            (9, 10, update![0, 0, -1]),
            (10, 11, update![0, 0, 0, -1]),
            (11, 3, update![Upd::Min(3)]),
            (10, 12, update![0, 0, -1]),
            (12, 10, update![0, 0, -1]),
            (0, 13, update![Upd::Min(2)]),
            (13, 14, update![-1]),
            (14, 15, update![0, -1]),
            (14, 16, update![0, 0, -1]),
            (16, 17, update![0, 0, -1]),
        ],
        &attacker_pos,
    );
    EnergyGame::standard_reach(graph)
}

fn combinations() -> EnergyGame {
    let attacker_pos: Vec<bool> = (0..20)
        .map(|i| [1, 2, 3, 4, 6, 10, 16].contains(&i))
        .collect();
    let graph = GameGraph::new(
        20,
        &[
            (0, 1, update![-1]),
            (0, 2, update![0, -1]),
            (0, 3, update![0, 0, 0, -1]),
            (0, 4, update![0, 0, 0, 0, 0, -1]),

            (1, 5, update![-1]),
            (5, 6, update![-1]),
            (6, 7, update![0, -1]),
            (6, 8, update![-1, 0]),
            (5, 10, update![Upd::Min(3)]),
            (1, 9, update![0, 0, -1]),
            (9, 10, update![0, 0, -1]),
            (10, 11, update![-1]),

            (2, 12, update![0, 0, -1]),
            (2, 13, update![0, -1]),

            (3, 14, update![0, 0, -1]),
            (3, 15, update![0, 0, 0, Upd::Min(5)]),
            (15, 16, update![0, 0, 0, -1]),
            (16, 17, update![0, 0, 0, -1]),
            (3, 18, update![0, -1]),

            (4, 19, update![0, 0, 0, 0, 0, -1]),
        ],
        &attacker_pos,
    );
    EnergyGame::standard_reach(graph)
}

async fn example() {
    let mut game = combinations();
    //let f = File::create("graph.json").unwrap();
    //serde_json::to_writer_pretty(f, &game.graph).unwrap();
    let energies = game.run().await.unwrap();
    println!("{:#?}", energies);
}

async fn run_json_graph() -> io::Result<()> {
    let path = env::args_os().nth(1).ok_or(io::Error::new(
            io::ErrorKind::Other, "Missing path argument"))?;
    let reader = File::open(path)?;
    let graph = serde_json::from_reader(&reader)?;
    let mut game = EnergyGame::standard_reach(graph);
    //let energies = game.run().await
    //    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut runner = game.get_gpu_runner().await.unwrap();
    runner.execute_gpu().await.unwrap();
    let energies = game.energies;
    println!("{:?}", energies);
    Ok(()) 
}

fn main() -> io::Result<()> {
    env_logger::init();
    let mut args = env::args_os();
    match args.len() {
        1 => Ok(pollster::block_on(example())),
        2 => pollster::block_on(run_json_graph()),
        _ => {
            eprintln!("Invalid arguments. Usage: {:?} [file]", args.next().unwrap_or_default());
            std::process::exit(2);
        },
    }
}
