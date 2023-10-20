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
    energy_offset: u32,
    sup_offset: u32,
}

var<workgroup> wg_node_offset: u32;

// Apply the update to e backwards
fn inv_update(e: u32, upd: u32) -> u32 {
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

fn find_start_node_idx(i: u32, l_idx: u32) -> u32 {
    let n_nodes = arrayLength(&node_offsets);
    let first_idx = i & (u32(-1i) << 6u); // Index of first element in workgroup
    let len_log64 = (firstLeadingBit(n_nodes - 1u) / 6u) + 1u;
    for (var stride = len_log64; stride > 0u; stride--) {
        let stride_width = 1u << stride * 6u; // 64**stride
        let search_offset = wg_node_offset + l_idx * stride_width;
        let search_max = min(search_offset + stride_width, n_nodes);
        if (search_offset <= search_max
            && node_offsets[search_offset].energy_offset <= first_idx
            && first_idx < node_offsets[search_max].energy_offset)
        {
            wg_node_offset = search_offset;
        }
    }

    for (var node_idx = wg_node_offset; node_idx < wg_node_offset + 64u; node_idx++) {
        if node_offsets[node_idx].energy_offset <= i && i < node_offsets[node_idx + 1u].energy_offset {
            return node_idx;
        }
    }
    // Couldn't find index. This should not happen.
    return u32(-1i);
}

// First part of the defence process: Only update all energies
@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g_id:vec3<u32>,
        @builtin(local_invocation_index) l_idx: u32)
{
    let i = g_id.x;
    let n_nodes = arrayLength(&node_offsets);

    let start_node_idx = find_start_node_idx(i, l_idx);

    if i >= node_offsets[n_nodes - 1u].energy_offset {
        return;
    }

    // For now the values here are repeated so the array has the same length as energies
    // Maybe the thing above should be done for this array as well.
    // successor_offsets holds the successor indices from each node's adjacency
    // list.
    let end_node = successor_offsets[i];
    let start_node = node_offsets[start_node_idx].node;

    // Update energies
    var update = graph_weights[graph_row_offsets[start_node] + end_node];

    let updated = inv_update(energies[i], update);
    energies[i] = updated;
}
