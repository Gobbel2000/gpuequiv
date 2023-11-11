@group(0) @binding(0)
var<storage> energies: array<u32>;

@group(0) @binding(1)
var<storage> node_offsets: array<NodeOffset>;

@group(0) @binding(2)
var<storage> successor_offsets: array<u32>;

@group(0) @binding(3)
var<storage, read_write> minima: array<u32>;

@group(1) @binding(0)
var<storage, read_write> suprema: array<u32>;


struct NodeOffset {
    node: u32,
    successor_offsets_idx: u32,
    energy_offset: u32,
    sup_offset: u32,
}

// Temporary buffer to hold the output flags as u32. This is later condensed
// into bit-packed u32's in the minima storage buffer.
var<workgroup> minima_buf: array<u32,64u>;
var<workgroup> wg_node: u32;

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

// This time with regards to sup_offset, not energy_offset
fn find_start_node_idx(i: u32, l_idx: u32) -> u32 {
    let n_nodes = arrayLength(&node_offsets);
    let first_idx = i & (u32(-1i) << 6u); // Index of first element in workgroup
    let len_log64 = (firstLeadingBit(n_nodes - 1u) / 6u);
    for (var stride = 1u << (len_log64 * 6u); stride > 0u; stride >>= 6u) {
        let search_offset = wg_node + l_idx * stride;
        let search_max = min(search_offset + stride, n_nodes - 1u);
        if (search_offset <= search_max
            && node_offsets[search_offset].sup_offset <= first_idx
            && first_idx < node_offsets[search_max].sup_offset)
        {
            wg_node = search_offset;
        }
        // Ensure wg_node is properly written
        workgroupBarrier();
    }

    for (var node_idx = wg_node; node_idx < wg_node + 64u; node_idx++) {
        if i >= node_offsets[node_idx].sup_offset && i < node_offsets[node_idx + 1u].sup_offset {
            return node_idx;
        }
    }
    // Couldn't find index. This should not happen.
    return u32(-1i);
}


@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g_id:vec3<u32>,
        @builtin(local_invocation_index) l_idx: u32)
{
    let i = g_id.x;
    let n_nodes = arrayLength(&node_offsets);

    let start_node_idx = find_start_node_idx(i, l_idx);

    if i >= node_offsets[n_nodes - 1u].sup_offset {
        return;
    }

    let node = node_offsets[start_node_idx];
    let node_idx = i - node.sup_offset;
    var combination = node_idx;

    var supremum = array<vec4<u32>,2u>(vec4(0u), vec4(0u));
    var energy_idx = node.energy_offset;
    for (var suc = node.successor_offsets_idx;
         suc < node_offsets[start_node_idx + 1u].successor_offsets_idx;
         suc++)
    {
        let suc_width = successor_offsets[suc + 1u] - successor_offsets[suc];
        let pick_idx = energy_idx + (combination % suc_width);
        let pick = energies[pick_idx];
        let energy = array<vec4<u32>,2u>(
            (vec4(pick) >> vec4(0u, 2u, 4u, 6u)) & vec4(0x3u),
            (vec4(pick) >> vec4(8u, 10u, 12u, 14u)) & vec4(0x3u),
        );
        supremum[0u] = max(supremum[0u], energy[0u]);
        supremum[1u] = max(supremum[1u], energy[1u]);
        energy_idx += suc_width;
        combination /= suc_width;
    }
    // Pack supremum into u32
    supremum[0u] = supremum[0u] << vec4(0u, 2u, 4u, 6u) |
        supremum[1u] << vec4(8u, 10u, 12u, 16u);
    let sup_packed = supremum[0u].x | supremum[0u].y | supremum[0u].z | supremum[0u].w;
    
    suprema[i] = sup_packed;
}

@compute
@workgroup_size(64, 1, 1)
fn minimize(@builtin(global_invocation_id) g_id: vec3<u32>,
            @builtin(local_invocation_index) l_idx: u32)
{
    let i = g_id.x;
    let n_nodes = arrayLength(&node_offsets);

    let start_node_idx = find_start_node_idx(i, l_idx);

    if i >= node_offsets[n_nodes - 1u].sup_offset {
        return;
    }

    let packing_offset = l_idx & 0x1fu; // zero everything but last 5 bits, l_idx % 32
    var is_minimal = 1u << packing_offset;
    let energy = suprema[i];

    let e_start = node_offsets[start_node_idx].sup_offset;
    let e_end = node_offsets[start_node_idx + 1u].sup_offset; // exclusive
    for (var j: u32 = e_start; j < e_end; j++) {
        let e2 = suprema[j];
        // Skip reflexive comparisons,
        // When energies are equal, keep only those with higher index
        if j != i && ((e2 == energy && i < j) ||
                      (e2 != energy && less_eq(e2, energy))) {
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
        minima[i >> 5u] = minima_buf[l_idx];
    }
}
