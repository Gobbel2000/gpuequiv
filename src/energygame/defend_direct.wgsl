alias Energy = array<u32, $ENERGY_SIZE>;

@group(0) @binding(0)
var<storage, read_write> energies: array<Energy>;

@group(0) @binding(1)
var<storage> node_offsets: array<NodeOffset>;

@group(0) @binding(2)
var<storage> successor_offsets: array<u32>;

@group(1) @binding(0)
var<storage, read_write> suprema: array<Energy>;

@group(1) @binding(1)
var<storage, read_write> minima: array<u32>;


struct NodeOffset {
    node: u32,
    successor_offsets_idx: u32,
    energy_offset: u32,
    sup_offset: u32,
}

// Temporary buffer to hold the output flags as u32. This is later condensed
// into bit-packed u32's in the minima storage buffer.
var<workgroup> minima_buf: array<u32,64u>;

fn less_eq(a: Energy, b: Energy) -> bool {
    $IMPL_LESS_EQ;
}

fn energy_eq(a: Energy, b: Energy) -> bool {
    $IMPL_EQ;
}

// This time with regards to sup_offset, not energy_offset
fn binsearch(i: u32) -> u32 {
    let last = arrayLength(&node_offsets) - 1u;
    var stride = arrayLength(&node_offsets);
    var l = 0u;
    while stride > 0u {
        // Halve stride, but don't overshoot buffer bounds
        stride = min(stride >> 1u, last - l);
        l += select(0u, stride + 1u,
                    node_offsets[l + stride].sup_offset <= i);
    }
    return l - 1u;
}


@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let i = g_id.x;
    let n_nodes = arrayLength(&node_offsets);

    if i >= node_offsets[n_nodes - 1u].sup_offset {
        return;
    }

    let start_node_idx = binsearch(i);
    let node = node_offsets[start_node_idx];
    let node_idx = i - node.sup_offset;
    var combination = node_idx;

    var supremum = array<u32, $ENERGY_ELEMENTS>();
    var energy_idx = node.energy_offset;
    for (var suc = node.successor_offsets_idx;
         suc < node_offsets[start_node_idx + 1u].successor_offsets_idx;
         suc++)
    {
        let suc_width = successor_offsets[suc + 1u] - successor_offsets[suc];
        let pick_idx = energy_idx + (combination % suc_width);
        let e = energies[pick_idx];

        let energy = $UNPACK_ENERGY;

        //supremum[0u] = max(supremum[0u], energy[0u]);
        //supremum[1u] = max(supremum[1u], energy[1u]);
        //...
        $MAX_SUPREMUM

        energy_idx += suc_width;
        combination /= suc_width;
    }
    // Pack supremum into u32
    suprema[i] = $PACK_SUPREMUM;
}

@compute
@workgroup_size(64, 1, 1)
fn minimize(@builtin(global_invocation_id) g_id: vec3<u32>,
            @builtin(local_invocation_index) l_idx: u32)
{
    let i = g_id.x;
    let n_nodes = arrayLength(&node_offsets);

    if i >= node_offsets[n_nodes - 1u].sup_offset {
        return;
    }

    let start_node_idx = binsearch(i);
    let packing_offset = l_idx & 0x1fu; // zero everything but last 5 bits, l_idx % 32
    var is_minimal = 1u << packing_offset;
    let energy = suprema[i];

    let e_start = node_offsets[start_node_idx].sup_offset;
    let e_end = node_offsets[start_node_idx + 1u].sup_offset; // exclusive
    for (var j: u32 = e_start; j < e_end; j++) {
        let e2 = suprema[j];
        // Skip reflexive comparisons,
        // When energies are equal, keep only those with higher index
        let eq = energy_eq(e2, energy);
        if j != i && ((eq && i < j) ||
                      (!eq && less_eq(e2, energy))) {
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
