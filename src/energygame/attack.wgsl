alias Energy = array<u32, $ENERGY_SIZE>;
alias Update = array<u32, $UPDATE_SIZE>;

@group(0) @binding(0)
var<storage, read_write> energies: array<Energy>;

@group(0) @binding(1)
var<storage> node_offsets: array<NodeOffset>;

@group(0) @binding(2)
var<storage> successor_offsets: array<u32>;

@group(0) @binding(3)
var<storage, read_write> minima: array<u32>;

@group(0) @binding(4)
var<uniform> work_size: u32;


@group(1) @binding(0)
var<storage> graph_row_offsets: array<u32>;

@group(1) @binding(1)
var<storage> graph_weights: array<Update>;

struct NodeOffset {
    node: u32,
    successor_offsets_idx: u32,
    offset: u32,
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

// Apply the update to e backwards
fn inv_update(e: Energy, u: Update) -> Energy {
    // Expand bit fields into full u32's
    var energy = $UNPACK_ENERGY;
    let updates = $UNPACK_UPDATE;

    // Apply 1-updates first
$UPDATE1

    // Look for min-updates
    // 0u means no update, 1 means 1-update, everything else represents
    // the second component in the min-operation, the first being the
    // current position i. To make place for the 2 special values, 2
    // must be subtracted here.
    $UPDATE_MIN

    return $PACK_ENERGY;
}

fn binsearch(i: u32) -> u32 {
    let last = work_size - 1u;
    var stride = work_size;
    var l = 0u;
    while stride > 0u {
        // Halve stride, but don't overshoot buffer bounds
        stride = min(stride >> 1u, last - l);
        l += select(0u, stride + 1u,
                    node_offsets[l + stride].offset <= i);
    }
    return l - 1u;
}


@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let i = g_id.x;
    if i >= node_offsets[work_size - 1u].offset {
        return;
    }

    let start_node_idx = binsearch(i);
    let node = node_offsets[start_node_idx];
    let start_node = node.node;

    // Update energies
    var update = array<u32, $UPDATE_SIZE>();
    // Search for successor to use
    for (var suc = node.successor_offsets_idx;
         // Stop before last entry, which correspond to the node's own energies.
         // `update` is left at 0 in that case.
         suc < (node_offsets[start_node_idx + 1u].successor_offsets_idx - 1u);
         suc++)
    {
        if i >= successor_offsets[suc] && i < successor_offsets[suc + 1u] {
            // Index of successor in adjacency list as well as weight array
            let successor_idx = suc - node.successor_offsets_idx;
            update = graph_weights[graph_row_offsets[start_node] + successor_idx];
            break;
        }
    }

    energies[i] = inv_update(energies[i], update);
}

@compute
@workgroup_size(64, 1, 1)
fn minimize(@builtin(global_invocation_id) g_id: vec3<u32>,
            @builtin(local_invocation_index) l_idx: u32)
{
    let i = g_id.x;
    if i >= node_offsets[work_size - 1u].offset {
        return;
    }

    let start_node_idx = binsearch(i);

    let packing_offset = l_idx & 0x1fu; // zero everything but last 5 bits, l_idx % 32
    var is_minimal = 1u << packing_offset;
    let energy = energies[i];

    let e_start = node_offsets[start_node_idx].offset;
    let e_end = node_offsets[start_node_idx + 1u].offset; // exclusive
    for (var j: u32 = e_start; j < e_end; j++) {
        let e2 = energies[j];
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
