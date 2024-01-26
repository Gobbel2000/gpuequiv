use std::ffi::OsStr;
use std::fs::File;
use std::env;
use std::io;
use std::ops::Deref;

use gpuequiv::*;
use gpuequiv::energygame::*;
use gpuequiv::gamebuild::*;

// Varied, multidimensional updates, leading to multidimensional energies
fn _multidimensional() -> EnergyGame {
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

fn combinations() -> EnergyGame {
    let conf = EnergyConf::STANDARD;
    let attacker_pos: Vec<bool> = (0..20)
        .map(|i| [1, 2, 3, 4, 6, 10, 16].contains(&i))
        .collect();
    let graph = GameGraph::new(
        20,
        vec![
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
        attacker_pos,
        conf,
    );
    EnergyGame::standard_reach(graph)
}

async fn energy_game() {
    let mut game = combinations();
    //let f = File::create("graph.json").unwrap();
    //serde_json::to_writer_pretty(f, &game.graph).unwrap();
    let energies = game.run().await.unwrap();
    for node in energies {
        println!("{}", node);
    }
}

async fn build_game() -> io::Result<()> {
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
    let mut builder = gamebuild::GameBuild::with_lts(lts);
    builder.compare(0, 1);
    let f = File::create("built_graph.json").unwrap();
    serde_json::to_writer_pretty(f, &builder.game).unwrap();
    for (i, p) in builder.nodes.iter().enumerate() {
        println!("{i}: \t{p}");
    }
    Ok(())
}

async fn run_json_graph() -> io::Result<()> {
    let path = env::args_os().nth(1).ok_or(io::Error::new(
            io::ErrorKind::Other, "Missing path argument"))?;
    let reader = File::open(path)?;
    let graph = serde_json::from_reader(&reader)?;
    let mut game = EnergyGame::standard_reach(graph);
    let energies = game.run().await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    for node in energies {
        println!("{}", node);
    }
    Ok(()) 
}

async fn all() -> io::Result<()> {
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
    for e in lts.winning_budgets(0, 1).await.unwrap() {
        println!("{e}");
    }
    Ok(())
}

async fn csv_lts(fname: &OsStr) -> io::Result<()> {
    let lts = TransitionSystem::from_csv_file(fname)?;
    let mut builder = GameBuild::with_lts(lts);
    let n_starting_points = builder.compare_all_but_bisimilar();
    println!("Game built");
    println!("Number of nodes: {}", builder.game.n_vertices());
    println!("Number of edges: {}", builder.game.column_indices.len());
    let nodes = builder.nodes;
    let mut energy_game = EnergyGame::standard_reach(builder.game);
    println!("Running game...");
    let energies = energy_game.run().await.unwrap();
    println!("Energy game finished");

    let mut equivalence_classes: Vec<Vec<u32>> = Vec::new();
    for (comp, e) in nodes[..n_starting_points as usize].iter().zip(energies) {
        let pos = match Deref::deref(comp) {
            Position::Attack(pos) => pos,
            _ => unreachable!(),
        };
        let class_p = match equivalence_classes.iter().position(|c| c.contains(&pos.p)) {
            Some(idx) => idx,
            None => {
                equivalence_classes.push(vec![pos.p]);
                equivalence_classes.len() - 1
            },
        };
        let q = pos.q[0];
        let class_q = equivalence_classes.iter().position(|c| c.contains(&q));
        if e.test_equivalence(std_equivalences::bisimulation()) {
            if let Some(cq) = class_q {
                if class_p != cq {
                    let removed = equivalence_classes.remove(class_p.max(cq));
                    equivalence_classes[class_p.min(cq)].extend(removed);
                }
            } else {
                equivalence_classes[class_p].push(q);
            }
        } else if class_q.is_none() {
            equivalence_classes.push(vec![q]);
        }
    }

    for c in &equivalence_classes {
        println!("{:?}", c);
    }
    println!("{}", equivalence_classes.len());
    Ok(())
}

fn csv_bisimulation(fname: &OsStr) -> io::Result<()> {
    let lts = TransitionSystem::from_csv_file(fname)?;
    let partition = lts.signature_refinement();
    let n_classes = partition.iter().max().copied().unwrap_or_default() + 1;
    let mut classes = vec![vec![]; n_classes as usize];
    for (i, &p) in partition.iter().enumerate() {
        classes[p as usize].push(i);
    }
    println!("{:?}", classes);
    println!("{n_classes}");
    Ok(())
}

fn main() -> io::Result<()> {
    env_logger::init();
    let mut args = env::args_os();
    match args.len() {
        1 => Ok(pollster::block_on(energy_game())),
        2 => match args.nth(1).unwrap().to_str() {
            Some("build") => pollster::block_on(build_game()),
            Some("all") => pollster::block_on(all()),
            _ => pollster::block_on(run_json_graph()),
        },
        3 => match args.nth(1).unwrap().to_str() {
            Some("csv") => pollster::block_on(csv_lts(&args.next().unwrap())),
            Some("bisim") => csv_bisimulation(&args.next().unwrap()),
            _ => Ok(invalid_args()),
        },
        _ => Ok(invalid_args()),
    }
}

fn invalid_args() {
    eprintln!("Invalid arguments. Usage: {:?} [file]", env::args_os().next().unwrap_or_default());
    std::process::exit(2);
}
