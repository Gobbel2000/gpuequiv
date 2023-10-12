@group(0) @binding(0)
var<storage, read_write> energies: array<u32>;

@group(0) @binding(1)
var<storage> node_offsets: array<NodeOffset>;

@group(0) @binding(2)
var<storage> successor_offsets: array<u32>;

@group(0) @binding(3)
var<storage, read_write> minima: array<u32>;


@group(1) @binding(0)
var<storage> graph_column_indices: array<u32>;

@group(1) @binding(1)
var<storage> graph_row_offsets: array<u32>;

@group(1) @binding(2)
var<storage> graph_weights: array<u32>;

struct NodeOffset {
    node: u32,
    offset: u32,
}

// Temporary buffer to hold the output flags as u32. This is later condensed
// into bit-packed u32's in the minima storage buffer.
var<workgroup> minima_buf: array<u32,64u>;
var<workgroup> wg_node_offset: u32;

fn less_eq(a: u32, b: u32) -> bool {
    return ((a & 0x3u) <= (b & 0x3u)
        && (a & 0xcu) <= (b & 0xcu)
        && (a & 0x30u) <= (b & 0x30u)
        && (a & 0xc0u) <= (b & 0xc0u)
        && (a & 0x300u) <= (b & 0x300u)
        && (a & 0xc00u) <= (b & 0xc00u)
        /* for 8-tuple energies
        && (a & 0x3000u) <= (b & 0x3000u)
        && (a & 0xc000u) <= (b & 0xc000u)
        */
    );
}

// Apply the update to e backwards
fn inv_update(e: u32, upd: u32) -> u32 {
    /*
    if upd == 0u {
        return e;
    }
    */
    // Expand bit fields into full u32's
    var energy = array<vec4<u32>,2u>(
        (vec4(e) >> vec4(0u, 2u, 4u, 6u)) & vec4(0x3u),
        (vec4(e) >> vec4(8u, 10u, 12u, 14u)) & vec4(0x3u),
    );

    let updates = array<vec4<u32>,2u>(
        (vec4(upd) >> vec4(0u, 4u, 8u, 12u)) & vec4(0xfu),
        (vec4(upd) >> vec4(16u, 20u, 24u, 28u)) & vec4(0xfu),
    );

    // Apply 1u-updates first
    energy[0u] += vec4<u32>(updates[0u] == 1u); // 1 encodes 1-update
    energy[1u] += vec4<u32>(updates[1u] == 1u);

    // Look for min-updates
    // 0u means no update, 1 means 1-update, everything else represents
    // the second component in the min-operation, the first being the
    // current position i. To make place for the 2 special values, 2
    // must be subtracted here.
    /*
    for (var i: u32 = 0u; i < 6u; i++) {
        var u = updates[i >> 2u][i & 0x3u];
        if u > 1u {
            u -= 2u;
            energy[u >> 2u][u & 0x3u] = max(energy[u >> 2u][u & 0x3u],
                                            energy[i >> 2u][i & 0x3u]);
        }
    }
    */

    // Unrolled loop
    var u = updates[0][0];
    if u > 1u {
        u -= 2u;
        energy[u >> 2u][u & 0x3u] = max(energy[u >> 2u][u & 0x3u], energy[0][0]);
    }
    u = updates[0][1];
    if u > 1u {
        u -= 2u;
        energy[u >> 2u][u & 0x3u] = max(energy[u >> 2u][u & 0x3u], energy[0][1]);
    }
    u = updates[0][2];
    if u > 1u {
        u -= 2u;
        energy[u >> 2u][u & 0x3u] = max(energy[u >> 2u][u & 0x3u], energy[0][2]);
    }
    u = updates[0][3];
    if u > 1u {
        u -= 2u;
        energy[u >> 2u][u & 0x3u] = max(energy[u >> 2u][u & 0x3u], energy[0][3]);
    }
    u = updates[1][0];
    if u > 1u {
        u -= 2u;
        energy[u >> 2u][u & 0x3u] = max(energy[u >> 2u][u & 0x3u], energy[1][0]);
    }
    u = updates[1][1];
    if u > 1u {
        u -= 2u;
        energy[u >> 2u][u & 0x3u] = max(energy[u >> 2u][u & 0x3u], energy[1][1]);
    }

    energy[0u] = min(energy[0u], vec4(3u)) << vec4(0u, 2u, 4u, 6u) |
                 min(energy[1u], vec4(3u)) << vec4(8u, 10u, 12u, 16u);
    return energy[0u].x | energy[0u].y | energy[0u].z | energy[0u].w;
}


@compute
@workgroup_size(64, 1, 1)
fn process_energies(@builtin(global_invocation_id) g_id: vec3<u32>,
          @builtin(local_invocation_index) l_idx: u32) {
    let i = g_id.x;
    let n_nodes = arrayLength(&node_offsets);
    if i >= node_offsets[n_nodes - 1u].offset {
        return;
    }

    // Gather graph start and end node corresponding to current energy
    let len_log64 = (firstLeadingBit(n_nodes - 1u) / 6u) + 1u;
    for (var stride = len_log64; stride > 0u; stride--) {
        let stride_width = 1u << stride * 6u; // l_idx * 64**stride
        let search_offset = wg_node_offset + l_idx * stride_width;
        let search_max = min(search_offset + stride_width, n_nodes);
        if (search_offset <= search_max
            && node_offsets[search_offset].offset <= l_idx
            && l_idx < node_offsets[search_max].offset)
        {
            wg_node_offset = search_offset;
        }
    }
    var start_node_idx = 0u;
    for (var node_idx = wg_node_offset; true; node_idx++) {
        if node_offsets[node_idx].offset <= l_idx && l_idx < node_offsets[node_idx + 1u].offset {
            start_node_idx = node_idx;
            break;
        }
    }

    // For now the values here are repeated so the array has the same length as energies
    // Maybe the thing above should be done for this array as well.
    // successor_offsets holds the successor indices from each node's adjacency
    // list.
    let end_node = successor_offsets[i];
    let start_node = node_offsets[start_node_idx].node;

    // Update energies
    let update = graph_weights[graph_row_offsets[start_node] + end_node];

    let updated = inv_update(energies[i], update);
    energies[i] = updated;

    workgroupBarrier();

    // Find minmal energies
    let packing_offset = l_idx & 0x1fu; // zero everything but last 5 bits, l_idx % 32
    var is_minimal = 1u << packing_offset;

    let e_start = node_offsets[start_node_idx].offset;
    let e_end = node_offsets[start_node_idx + 1u].offset; // exclusive
    let e_idx = i - e_start; // Index within chunk of energies to compare
    for (var j: u32 = e_start; j < e_end; j++) {
        let e2 = energies[j];
        // Skip reflexive comparisons,
        // When energies are equal, keep only those with higher index
        if j != e_idx && ((e2 == updated && e_idx < j) ||
                          (e2 != updated && less_eq(e2, updated))) {
            // Mark to be filtered out
            is_minimal = 0u;
            break;
        }
    }
    minima_buf[l_idx] = is_minimal;

    workgroupBarrier();

    // Reduce all 64 flags into 2 bit-packed u32
    // Valid results only end up in the first elements of both 32-blocks.
    // The other threads just do the same because it is faster than adding conditions.
    // This is also a case of loop-unrolling.
    minima_buf[l_idx] |= minima_buf[l_idx + 16u];
    minima_buf[l_idx] |= minima_buf[l_idx + 8u];
    minima_buf[l_idx] |= minima_buf[l_idx + 4u];
    minima_buf[l_idx] |= minima_buf[l_idx + 2u];
    minima_buf[l_idx] |= minima_buf[l_idx + 1u];

    // The first thread in both 32-blocks writes the packed u32 out
    if packing_offset == 0u {
        minima[g_id.x >> 5u] = minima_buf[l_idx];
    }
}
