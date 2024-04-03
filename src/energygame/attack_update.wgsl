// This shader file gets manipulated in a pre-processing step.
//
// Any identifiers starting with a `$` are replaced with proper
// code before compilation, for example `$ENERGY_SIZE`.
// The replacements are defined in the file `shadergen.rs`.
//
// Pre-processed shader files can be inspected by running with the
// environment variable `GPUEQUIV_DUMP=1`.
// Doing so writes pre-processed shaders into a new directory `./shaders_dump`.

alias Energy = array<u32, $ENERGY_SIZE>;
alias Update = array<u32, $UPDATE_SIZE>;

@group(0) @binding(0)
var<storage, read_write> energies: array<Energy>;

@group(0) @binding(1)
var<storage> node_offsets: array<NodeOffset>;

@group(0) @binding(2)
var<storage> successor_offsets: array<u32>;

@group(0) @binding(3)
var<storage, read_write> changed: array<u32>;

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
fn update(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let i = g_id.x;
    if i < arrayLength(&changed) {
        // Zero `changed` array, so it can always be assumed to be initially 0
        // in attack_minimize.wgsl. Doing it here already ensures global
        // synchronization until then.
        changed[i] = 0u;
    }
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
