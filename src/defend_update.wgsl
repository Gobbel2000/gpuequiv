alias Energy = array<u32, $ENERGY_SIZE>;
alias Update = array<u32, $UPDATE_SIZE>;

@group(0) @binding(0)
var<storage, read_write> energies: array<Energy>;

@group(0) @binding(1)
var<storage> node_offsets: array<NodeOffset>;

@group(0) @binding(2)
var<storage> successor_offsets: array<u32>;


@group(1) @binding(0)
var<storage> graph_column_indices: array<u32>;

@group(1) @binding(1)
var<storage> graph_row_offsets: array<u32>;

@group(1) @binding(2)
var<storage> graph_weights: array<Update>;

struct NodeOffset {
    node: u32,
    successor_offsets_idx: u32,
    energy_offset: u32,
    sup_offset: u32,
}

var<workgroup> wg_node: u32;

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

fn find_start_node_idx(i: u32, l_idx: u32) -> u32 {
    let n_nodes = arrayLength(&node_offsets);
    let first_idx = i & (u32(-1i) << 6u); // Index of first element in workgroup
    let len_log64 = (firstLeadingBit(n_nodes - 1u) / 6u);
    for (var stride = 1u << (len_log64 * 6u); stride > 0u; stride >>= 6u) {
        let search_offset = wg_node + l_idx * stride;
        let search_max = min(search_offset + stride, n_nodes - 1u);
        if (search_offset <= search_max
            && node_offsets[search_offset].energy_offset <= first_idx
            && first_idx < node_offsets[search_max].energy_offset)
        {
            wg_node = search_offset;
        }
        // Ensure wg_node is properly written
        workgroupBarrier();
    }

    for (var node_idx = wg_node; node_idx < wg_node + 64u; node_idx++) {
        if i >= node_offsets[node_idx].energy_offset && i < node_offsets[node_idx + 1u].energy_offset {
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

    let node = node_offsets[start_node_idx];
    let start_node = node.node;

    // Update energies
    var update = array<u32, $UPDATE_SIZE>();
    // Search for successor to use
    for (var suc = node.successor_offsets_idx;
         suc < node_offsets[start_node_idx + 1u].successor_offsets_idx;
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
