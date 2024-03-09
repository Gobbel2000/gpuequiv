// Example to show JSON serialization and deserialization of energy game graphs.
//
// When run with no arguments, a game graph is written to `./graph.json`.
// With a path as a command line argument (for example 'graph.json',
// that was previously written by this example), that file is read as
// the input game graph and processed.
//
// Example usage:
// `cargo run --example json`
//     (Writes a graph into ./graph.json)
// `cargo run --example json -- graph.json`
//     (Reads that graph again and processes it)

use std::ffi::OsStr;
use std::fs::File;
use std::env;
use std::io;

use gpuequiv::*;
use gpuequiv::energygame::*;

fn game() -> EnergyGame {
    let conf = EnergyConf::STANDARD;
    let attacker_pos: Vec<bool> = (0..18)
        .map(|i| [0, 2, 4, 6, 9, 11, 12, 14, 17].contains(&i))
        .collect();
    let graph = GameGraph::new(
        18,
        vec![
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
        attacker_pos,
        conf,
    );
    EnergyGame::standard_reach(graph)
}

fn dump_game() {
    let game = game();

    const FNAME: &str = "graph.json";
    let f = File::create("graph.json").unwrap();
    serde_json::to_writer_pretty(f, &game.graph).unwrap();

    println!("Written energy game graph to `./{FNAME}`");
    println!("Try processing it with `{} {FNAME}`.",
             env::args().next().unwrap_or_default());
}

async fn run_json_graph(fname: &OsStr) -> io::Result<()> {
    let reader = File::open(fname)?;
    let graph = serde_json::from_reader(&reader)?;
    println!("Read game graph from {}.", fname.to_string_lossy());

    // Create an energy game using the game graph
    let game = EnergyGame::standard_reach(graph);
    // Run the game on GPU
    let energies = game.run().await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    for node in energies {
        println!("{}", node);
    }
    Ok(()) 
}

fn main() -> io::Result<()> {
    env_logger::init();
    let mut args = env::args_os();
    if args.len() > 1 {
        pollster::block_on(run_json_graph(&args.nth(1).unwrap()))
    } else {
        Ok(dump_game())
    }
}
