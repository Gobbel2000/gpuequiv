@group(0) @binding(0)
var<storage, read_write> energies: array<u32>;

@group(0) @binding(1)
var<storage> node_offsets: array<NodeOffset>;

@group(0) @binding(2)
var<storage> sup_offsets: array<u32>;

@group(0) @binding(3)
var<storage, read_write> minima: array<u32>;


struct NodeOffset {
    node: u32,
    energy_offset: u32,
    sup_offset: u32,
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

// This time with regards to sup_offset, not energy_offset
fn find_start_node_idx(i: u32, l_idx: u32) -> u32 {
    let n_nodes = arrayLength(&node_offsets);
    let first_idx = i & (u32(-1i) << 6u); // Index of first element in workgroup
    let len_log64 = (firstLeadingBit(n_nodes - 1u) / 6u) + 1u;
    for (var stride = len_log64; stride > 0u; stride--) {
        let stride_width = 1u << stride * 6u; // 64**stride
        let search_offset = wg_node_offset + l_idx * stride_width;
        let search_max = min(search_offset + stride_width, n_nodes);
        if (search_offset <= search_max
            && node_offsets[search_offset].sup_offset <= first_idx
            && first_idx < node_offsets[search_max].sup_offset)
        {
            wg_node_offset = search_offset;
        }
    }

    for (var node_idx = wg_node_offset; node_idx < wg_node_offset + 64u; node_idx++) {
        if node_offsets[node_idx].sup_offset <= i && i < node_offsets[node_idx + 1u].sup_offset {
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

    
}
