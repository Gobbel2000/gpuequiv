use std::ffi::OsStr;
use std::env;
use std::io;
use std::ops::Deref;

use gpuequiv::*;
use gpuequiv::energygame::*;
use gpuequiv::gamebuild::*;

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
