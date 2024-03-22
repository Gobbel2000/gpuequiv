use super::*;

use bytemuck::Zeroable;

impl GPURunner<'_> {
    async fn test_def_shader(&self) -> Result<()> {
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Defend Intersection shader command encoder")
        });
        self.defiter_shader.compute_pass(&mut encoder, &self.graph_bind_group);

        // Submit command encoder for processing by GPU
        self.gpu.queue.submit(Some(encoder.finish()));

        const MAPS: usize = 2;
        let (sender, receiver) = channel(MAPS);
        self.defiter_shader.map_buffers(&sender);

        // Wait for the GPU to finish work
        self.gpu.device.poll(wgpu::Maintain::Wait);

        for _ in 0..MAPS {
            receiver.receive().await.expect("Channel should not be closed")?;
        }

        Ok(())
    }
}

fn game() -> (EnergyGame, EnergyArray) {
    let graph = GameGraph::new(
        5,
        vec![
            (0, 1, vec![-1]),
            (0, 2, vec![0, -1]),
            (0, 3, vec![0, 0, 0, -1]),
            (0, 4, vec![0, 0, 0, 0, 0, -1]),
        ],
        vec![false, true, true, true, true],
        EnergyConf::STANDARD,
    );
    let game = EnergyGame::with_reach(graph, vec![]);
    let energies = EnergyArray::from_conf([
            vec![2, 1, 1],
            vec![3, 0, 1],
            vec![1, 0, 2],

            vec![0, 0, 1],
            vec![0, 1],

            vec![0, 0, 0, 2, 2],
            vec![0, 0, 1],
            vec![0, 1],

            vec![0, 0, 0, 0, 0, 1],
        ].as_slice(),
        EnergyConf::STANDARD,
    ).unwrap();
    (game, energies)
}

#[pollster::test]
async fn test_defend_intersection_shader() {
    let (mut game, energies) = game();
    let mut runner = game.get_gpu_runner().await.unwrap();

    // Prepare input
    runner.defiter_shader.update(
        vec![NodeOffsetDef::zeroed(), NodeOffsetDef {
            node: u32::MAX,
            successor_offsets_idx: 4,
            energy_offset: 9,
            sup_offset: 64,
        }],
        vec![0, 3, 5, 8, 9],
        energies,
    );

    // Run shader
    runner.test_def_shader().await.unwrap();

    // Verify output
    let status_data = runner.defiter_shader.status_staging_buf.slice(..).get_mapped_range();
    let status: &[i32] = bytemuck::cast_slice(&status_data);

    let sup_data = runner.defiter_shader.sup_staging_buf.slice(..).get_mapped_range();
    let sup_vec: Vec<u32> = bytemuck::cast_slice(&sup_data).to_vec();
    let energy_size = runner.game.graph.get_conf().energy_size() as usize;
    let n_sup = sup_vec.len() / energy_size;
    let sup_array = ArrayView2::from_shape((n_sup, energy_size), &sup_vec).expect("Suprema array has invalid shape");

    println!("Defend Status values:\n{:?}", status);
    let suprema = EnergyArray::from_array(sup_array.to_owned(), runner.game.graph.get_conf());
    println!("Defend Suprema:\n{}", suprema);

    assert_eq!(status[0], 2);
    let earray = sup_array.slice(s![0..2, ..]);
    assert_eq!(earray, EnergyArray::from_conf([
                vec![3, 1, 1, 1, 0, 2],
                vec![2, 1, 2, 1, 0, 2],
            ].as_slice(),
            EnergyConf::STANDARD,
        ).unwrap(),
    );

    // Unmap buffers
    drop(status_data);
    runner.defiter_shader.status_staging_buf.unmap();
    drop(sup_data);
    runner.defiter_shader.sup_staging_buf.unmap();
}

#[pollster::test]
async fn test_defend_intersection_shader_large() {
    const NODES: u32 = 5;
    let (mut game, e_once) = game();
    let mut runner = game.get_gpu_runner().await.unwrap();

    let mut node_offsets = Vec::new();
    let mut successor_offsets = Vec::new();
    let mut energies = Array2::zeros((0, EnergyConf::STANDARD.energy_size() as usize));

    for i in 0..NODES + 1 {
        let sd = 9 * i;
        node_offsets.push(NodeOffsetDef {
            node: 0,
            successor_offsets_idx: 4 * i,
            energy_offset: sd,
            sup_offset: 64 * i,
        });
        successor_offsets.extend([sd, sd + 3, sd + 5, sd + 8]);
        energies.append(Axis(0), e_once.view()).unwrap();
    }

    runner.defiter_shader.update(
        node_offsets,
        successor_offsets,
        EnergyArray::from_array(energies, EnergyConf::STANDARD)
    );

    runner.test_def_shader().await.unwrap();

    // Verify output
    let status_data = runner.defiter_shader.status_staging_buf.slice(..).get_mapped_range();
    let status: &[i32] = bytemuck::cast_slice(&status_data);

    let sup_data = runner.defiter_shader.sup_staging_buf.slice(..).get_mapped_range();
    let sup_vec: Vec<u32> = bytemuck::cast_slice(&sup_data).to_vec();
    let energy_size = runner.game.graph.get_conf().energy_size() as usize;
    let n_sup = sup_vec.len() / energy_size;
    let sup_array = ArrayView2::from_shape((n_sup, energy_size), &sup_vec).expect("Suprema array has invalid shape");

    let node_offsets = &runner.defiter_shader.node_offsets;
    for (node, &status) in node_offsets[..node_offsets.len() - 1].iter().zip(status) {
        assert_eq!(status, 2);

        let start = node.sup_offset as usize;
        let end = start + status as usize;
        let earray = sup_array.slice(s![start..end, ..]);
        assert_eq!(earray, EnergyArray::from_conf([
                    vec![3, 1, 1, 1, 0, 2],
                    vec![2, 1, 2, 1, 0, 2],
                ].as_slice(),
                EnergyConf::STANDARD,
            ).unwrap(),
        );
    }

    // Unmap buffers
    drop(status_data);
    runner.defiter_shader.status_staging_buf.unmap();
    drop(sup_data);
    runner.defiter_shader.sup_staging_buf.unmap();
}

fn antichain() -> EnergyArray {
    // Antichain with 141 elements
    let energies = [
        vec![0, 0, 0, 2, 2, 2], vec![0, 0, 1, 1, 2, 2], vec![0, 0, 1, 2, 1, 2], vec![0, 0, 1, 2, 2, 1],
        vec![0, 0, 2, 0, 2, 2], vec![0, 0, 2, 1, 1, 2], vec![0, 0, 2, 1, 2, 1], vec![0, 0, 2, 2, 0, 2],
        vec![0, 0, 2, 2, 1, 1], vec![0, 0, 2, 2, 2, 0], vec![0, 1, 0, 1, 2, 2], vec![0, 1, 0, 2, 1, 2],
        vec![0, 1, 0, 2, 2, 1], vec![0, 1, 1, 0, 2, 2], vec![0, 1, 1, 1, 1, 2], vec![0, 1, 1, 1, 2, 1],
        vec![0, 1, 1, 2, 0, 2], vec![0, 1, 1, 2, 1, 1], vec![0, 1, 1, 2, 2, 0], vec![0, 1, 2, 0, 1, 2],
        vec![0, 1, 2, 0, 2, 1], vec![0, 1, 2, 1, 0, 2], vec![0, 1, 2, 1, 1, 1], vec![0, 1, 2, 1, 2, 0],
        vec![0, 1, 2, 2, 0, 1], vec![0, 1, 2, 2, 1, 0], vec![0, 2, 0, 0, 2, 2], vec![0, 2, 0, 1, 1, 2],
        vec![0, 2, 0, 1, 2, 1], vec![0, 2, 0, 2, 0, 2], vec![0, 2, 0, 2, 1, 1], vec![0, 2, 0, 2, 2, 0],
        vec![0, 2, 1, 0, 1, 2], vec![0, 2, 1, 0, 2, 1], vec![0, 2, 1, 1, 0, 2], vec![0, 2, 1, 1, 1, 1],
        vec![0, 2, 1, 1, 2, 0], vec![0, 2, 1, 2, 0, 1], vec![0, 2, 1, 2, 1, 0], vec![0, 2, 2, 0, 0, 2],
        vec![0, 2, 2, 0, 1, 1], vec![0, 2, 2, 0, 2, 0], vec![0, 2, 2, 1, 0, 1], vec![0, 2, 2, 1, 1, 0],
        vec![0, 2, 2, 2, 0, 0], vec![1, 0, 0, 1, 2, 2], vec![1, 0, 0, 2, 1, 2], vec![1, 0, 0, 2, 2, 1],
        vec![1, 0, 1, 0, 2, 2], vec![1, 0, 1, 1, 1, 2], vec![1, 0, 1, 1, 2, 1], vec![1, 0, 1, 2, 0, 2],
        vec![1, 0, 1, 2, 1, 1], vec![1, 0, 1, 2, 2, 0], vec![1, 0, 2, 0, 1, 2], vec![1, 0, 2, 0, 2, 1],
        vec![1, 0, 2, 1, 0, 2], vec![1, 0, 2, 1, 1, 1], vec![1, 0, 2, 1, 2, 0], vec![1, 0, 2, 2, 0, 1],
        vec![1, 0, 2, 2, 1, 0], vec![1, 1, 0, 0, 2, 2], vec![1, 1, 0, 1, 1, 2], vec![1, 1, 0, 1, 2, 1],
        vec![1, 1, 0, 2, 0, 2], vec![1, 1, 0, 2, 1, 1], vec![1, 1, 0, 2, 2, 0], vec![1, 1, 1, 0, 1, 2],
        vec![1, 1, 1, 0, 2, 1], vec![1, 1, 1, 1, 0, 2], vec![1, 1, 1, 1, 1, 1], vec![1, 1, 1, 1, 2, 0],
        vec![1, 1, 1, 2, 0, 1], vec![1, 1, 1, 2, 1, 0], vec![1, 1, 2, 0, 0, 2], vec![1, 1, 2, 0, 1, 1],
        vec![1, 1, 2, 0, 2, 0], vec![1, 1, 2, 1, 0, 1], vec![1, 1, 2, 1, 1, 0], vec![1, 1, 2, 2, 0, 0],
        vec![1, 2, 0, 0, 1, 2], vec![1, 2, 0, 0, 2, 1], vec![1, 2, 0, 1, 0, 2], vec![1, 2, 0, 1, 1, 1],
        vec![1, 2, 0, 1, 2, 0], vec![1, 2, 0, 2, 0, 1], vec![1, 2, 0, 2, 1, 0], vec![1, 2, 1, 0, 0, 2],
        vec![1, 2, 1, 0, 1, 1], vec![1, 2, 1, 0, 2, 0], vec![1, 2, 1, 1, 0, 1], vec![1, 2, 1, 1, 1, 0],
        vec![1, 2, 1, 2, 0, 0], vec![1, 2, 2, 0, 0, 1], vec![1, 2, 2, 0, 1, 0], vec![1, 2, 2, 1, 0, 0],
        vec![2, 0, 0, 0, 2, 2], vec![2, 0, 0, 1, 1, 2], vec![2, 0, 0, 1, 2, 1], vec![2, 0, 0, 2, 0, 2],
        vec![2, 0, 0, 2, 1, 1], vec![2, 0, 0, 2, 2, 0], vec![2, 0, 1, 0, 1, 2], vec![2, 0, 1, 0, 2, 1],
        vec![2, 0, 1, 1, 0, 2], vec![2, 0, 1, 1, 1, 1], vec![2, 0, 1, 1, 2, 0], vec![2, 0, 1, 2, 0, 1],
        vec![2, 0, 1, 2, 1, 0], vec![2, 0, 2, 0, 0, 2], vec![2, 0, 2, 0, 1, 1], vec![2, 0, 2, 0, 2, 0],
        vec![2, 0, 2, 1, 0, 1], vec![2, 0, 2, 1, 1, 0], vec![2, 0, 2, 2, 0, 0], vec![2, 1, 0, 0, 1, 2],
        vec![2, 1, 0, 0, 2, 1], vec![2, 1, 0, 1, 0, 2], vec![2, 1, 0, 1, 1, 1], vec![2, 1, 0, 1, 2, 0],
        vec![2, 1, 0, 2, 0, 1], vec![2, 1, 0, 2, 1, 0], vec![2, 1, 1, 0, 0, 2], vec![2, 1, 1, 0, 1, 1],
        vec![2, 1, 1, 0, 2, 0], vec![2, 1, 1, 1, 0, 1], vec![2, 1, 1, 1, 1, 0], vec![2, 1, 1, 2, 0, 0],
        vec![2, 1, 2, 0, 0, 1], vec![2, 1, 2, 0, 1, 0], vec![2, 1, 2, 1, 0, 0], vec![2, 2, 0, 0, 0, 2],
        vec![2, 2, 0, 0, 1, 1], vec![2, 2, 0, 0, 2, 0], vec![2, 2, 0, 1, 0, 1], vec![2, 2, 0, 1, 1, 0],
        vec![2, 2, 0, 2, 0, 0], vec![2, 2, 1, 0, 0, 1], vec![2, 2, 1, 0, 1, 0], vec![2, 2, 1, 1, 0, 0],
        vec![2, 2, 2, 0, 0, 0],
    ];
    EnergyArray::from_conf(energies.as_slice(), EnergyConf::STANDARD)
        .unwrap()
}

fn antichain2() -> EnergyArray {
    let ac = antichain();
    let conf = ac.get_conf();
    let array = ac.into_inner();
    let add = Energy::new(&[1; 6], conf).unwrap().raw_data().to_vec();
    assert_eq!(add.len(), 1);
    let array = array + add[0];
    EnergyArray::from_array(array, conf)
}

#[pollster::test]
async fn test_defend_intersection_antichain() {
    let conf = EnergyConf::STANDARD;
    let graph = GameGraph::new(
        5,
        vec![
            (0, 1, Update::zero(conf)),
            (0, 2, Update::zero(conf)),
            (0, 3, Update::zero(conf)),
            (0, 4, Update::zero(conf)),
        ],
        vec![false, true, true, true, true],
        conf,
    );
    let mut game = EnergyGame::with_reach(graph, vec![]);
    let mut runner = game.get_gpu_runner().await.unwrap();

    let ac = antichain();
    let n_energies = ac.n_energies() as u32;
    assert_eq!(n_energies, 141);
    let mut array = ac.into_inner();
    let ac2 = array.clone();
    // Repeat 3 more times (total 4)
    array.append(Axis(0), ac2.view()).unwrap();
    array.append(Axis(0), ac2.view()).unwrap();
    array.append(Axis(0), ac2.view()).unwrap();

    // Prepare input
    runner.defiter_shader.update(
        vec![NodeOffsetDef::zeroed(), NodeOffsetDef {
            node: u32::MAX,
            successor_offsets_idx: 4,
            energy_offset: n_energies * 4,
            sup_offset: 64,
        }],
        vec![0, n_energies, n_energies * 2, n_energies * 3, n_energies * 4],
        EnergyArray::from_array(array, conf),
    );

    // Run shader
    runner.test_def_shader().await.unwrap();

    // Verify output
    let status_data = runner.defiter_shader.status_staging_buf.slice(..).get_mapped_range();
    let status: &[i32] = bytemuck::cast_slice(&status_data);
    // Should fail due to insufficient memory
    assert_eq!(status[0], - (n_energies as i32));

    // Unmap buffers
    drop(status_data);
    runner.defiter_shader.status_staging_buf.unmap();
    runner.defiter_shader.sup_staging_buf.unmap();
    
    // A bit more memory this time: 256 energies
    runner.defiter_shader.update(
        vec![NodeOffsetDef::zeroed(), NodeOffsetDef {
            node: u32::MAX,
            successor_offsets_idx: 4,
            energy_offset: n_energies * 4,
            sup_offset: 256,
        }],
        vec![0, n_energies, n_energies * 2, n_energies * 3, n_energies * 4],
        runner.defiter_shader.energies.clone(),
    );

    // Run shader
    runner.test_def_shader().await.unwrap();

    // Verify output
    let status_data = runner.defiter_shader.status_staging_buf.slice(..).get_mapped_range();
    let status: &[i32] = bytemuck::cast_slice(&status_data);
    assert_eq!(status[0], -257);

    // Unmap buffers
    drop(status_data);
    runner.defiter_shader.status_staging_buf.unmap();
    runner.defiter_shader.sup_staging_buf.unmap();

    println!("Now with sufficiently large array");
    runner.defiter_shader.update(
        vec![NodeOffsetDef::zeroed(), NodeOffsetDef {
            node: u32::MAX,
            successor_offsets_idx: 4,
            energy_offset: n_energies * 4,
            sup_offset: 512,
        }],
        vec![0, n_energies, n_energies * 2, n_energies * 3, n_energies * 4],
        runner.defiter_shader.energies.clone(),
    );

    // Run shader
    runner.test_def_shader().await.unwrap();

    // Verify output
    let status_data = runner.defiter_shader.status_staging_buf.slice(..).get_mapped_range();
    let status: &[i32] = bytemuck::cast_slice(&status_data);
    let sup_data = runner.defiter_shader.sup_staging_buf.slice(..).get_mapped_range();
    let sup_vec: Vec<u32> = bytemuck::cast_slice(&sup_data).to_vec();
    let energy_size = runner.game.graph.get_conf().energy_size() as usize;
    let n_sup = sup_vec.len() / energy_size;
    let sup_array = ArrayView2::from_shape((n_sup, energy_size), &sup_vec).expect("Suprema array has invalid shape");

    // The final result should be exactly the antichain, since all successors have the same
    // energies
    assert_eq!(status[0], n_energies as i32);
    let earray = sup_array.slice(s![0..status[0], ..]);
    assert_eq!(earray, ac2);

    // Unmap buffers
    drop(status_data);
    runner.defiter_shader.status_staging_buf.unmap();
    drop(sup_data);
    runner.defiter_shader.sup_staging_buf.unmap();
}

// Run multiple, different nodes at once
#[pollster::test]
async fn test_defend_intersection_mixed() {
    let graph = GameGraph::new(
        15,
        vec![
            (0, 1, vec![-1]),
            (0, 2, vec![0, -1]),
            (0, 3, vec![0, 0, 0, -1]),
            (0, 4, vec![0, 0, 0, 0, 0, -1]),

            (5, 6, vec![0]),
            (5, 7, vec![0]),
            (5, 8, vec![0]),
            (5, 9, vec![0]),

            (10, 11, vec![0]),
            (10, 12, vec![0]),
            (10, 13, vec![0]),
            (10, 14, vec![0]),
        ],
        vec![false, true, true, true, true,
             false, true, true, true, true,
             false, true, true, true, true,
        ],
        EnergyConf::STANDARD,
    );

    let mut game = EnergyGame::with_reach(graph, vec![]);
    let mut runner = game.get_gpu_runner().await.unwrap();

    let energies = EnergyArray::from_conf([
            vec![2, 1, 1],
            vec![3, 0, 1],
            vec![1, 0, 2],

            vec![0, 0, 1],
            vec![0, 1],

            vec![0, 0, 0, 2, 2],
            vec![0, 0, 1],
            vec![0, 1],

            vec![0, 0, 0, 0, 0, 1],
        ].as_slice(),
        EnergyConf::STANDARD,
    ).unwrap();

    let ac = antichain();
    let ac2 = antichain2();
    let size_ac = ac.n_energies() as u32;
    assert_eq!(size_ac, 141);
    assert_eq!(ac.n_energies(), ac2.n_energies());

    let mut array = EnergyArray::empty(EnergyConf::STANDARD).into_inner();
    array.append(Axis(0), ac.view()).unwrap();
    array.append(Axis(0), ac.view()).unwrap();
    array.append(Axis(0), ac.view()).unwrap();
    array.append(Axis(0), ac.view()).unwrap();

    array.append(Axis(0), ac2.view()).unwrap();
    array.append(Axis(0), ac2.view()).unwrap();
    array.append(Axis(0), ac2.view()).unwrap();
    array.append(Axis(0), ac2.view()).unwrap();

    array.append(Axis(0), energies.view()).unwrap();

    let mut successor_offsets: Vec<u32> = (0..=8).map(|i| i * size_ac).collect();
    let last = *successor_offsets.last().unwrap();
    successor_offsets.extend([last + 3, last + 5, last + 8, last + 9]);

    // Prepare input
    runner.defiter_shader.update(
        vec![
            NodeOffsetDef {
                node: 5,
                successor_offsets_idx: 0,
                energy_offset: 0,
                sup_offset: 0,
            },
            NodeOffsetDef {
                node: 10,
                successor_offsets_idx: 4,
                energy_offset: size_ac * 4,
                sup_offset: 512,
            },
            NodeOffsetDef {
                node: 0,
                successor_offsets_idx: 8,
                energy_offset: size_ac * 4 * 2,
                sup_offset: 512 * 2,
            },
            NodeOffsetDef {
                node: u32::MAX,
                successor_offsets_idx: 12,
                energy_offset: size_ac * 4 * 2 + 9,
                sup_offset: 512 * 2 + 64,
            },
        ],
        successor_offsets,
        EnergyArray::from_array(array, EnergyConf::STANDARD),
    );

    // Run shader
    runner.test_def_shader().await.unwrap();

    // Verify output
    let status_data = runner.defiter_shader.status_staging_buf.slice(..).get_mapped_range();
    let status: &[i32] = bytemuck::cast_slice(&status_data);
    let sup_data = runner.defiter_shader.sup_staging_buf.slice(..).get_mapped_range();
    let sup_vec: Vec<u32> = bytemuck::cast_slice(&sup_data).to_vec();
    let energy_size = runner.game.graph.get_conf().energy_size() as usize;
    let n_sup = sup_vec.len() / energy_size;
    let sup_array = ArrayView2::from_shape((n_sup, energy_size), &sup_vec).expect("Suprema array has invalid shape");

    // The final result should be exactly the antichain, since all successors have the same
    // energies
    assert_eq!(status[0], size_ac as i32);
    assert_eq!(status[1], size_ac as i32);
    assert_eq!(status[2], 2);
    let earray0 = sup_array.slice(s![0..size_ac as usize, ..]);
    assert_eq!(earray0, ac);
    let earray1 = sup_array.slice(s![512..(512 + size_ac as usize), ..]);
    assert_eq!(earray1, ac2);
    let earray2 = sup_array.slice(s![1024..1026, ..]);
    assert_eq!(earray2, EnergyArray::from_conf([
                vec![3, 1, 1, 1, 0, 2],
                vec![2, 1, 2, 1, 0, 2],
            ].as_slice(),
            EnergyConf::STANDARD,
        ).unwrap(),
    );

    // Unmap buffers
    drop(status_data);
    runner.defiter_shader.status_staging_buf.unmap();
    drop(sup_data);
    runner.defiter_shader.sup_staging_buf.unmap();
}
