@group(0) @binding(0)
var<storage> clauses: array<Position>;

@group(0) @binding(1)
var<storage, read_write> atk_pos_out: array<Position>;


// Input labeled transition system in CSR format
@group(1) @binding(0)
var<storage> graph_column_indices: array<u32>;

@group(1) @binding(1)
var<storage> graph_row_offsets: array<u32>;

@group(1) @binding(2)
var<storage> graph_labels: array<u32>;


struct LinkedList {
    data: array<u32,4>,
    len: u32,
    next: u32,
}

struct Position {
    p: u32,
    Q: LinkedList,
}

@compute
@workgroup_size(64, 1, 1)
fn process_clauses(@builtin(global_invocation_id) g_id: vec3<u32>,
                   @builtin(local_invocation_index) l_idx: u32)
{
    let i = g_id.x;
    if i >= arrayLength(&clauses) {
        return;
    }

    let clause = clauses[i];

    atk_pos_out[i * 2u] = clause;
    atk_pos_out[i * 2u + 1u] = Position(
        clause.Q.data[0u],
        LinkedList (
            array<u32,4>(clause.p, 0u, 0u, 0u),
            1u,
            0u,
        ),
    );
}
