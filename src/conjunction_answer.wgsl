@group(0) @binding(0)
var<storage> input: array<ConjunctionPosition>;

@group(0) @binding(1)
var<storage, read_write> output: array<Position>;

// Dynamically allocated memory area for storing arbitrarily sized sets of
// processes alongside each position in the form of linked lists.
@group(0) @binding(2)
var<storage> heap_in: array<LinkedList>;


@group(1) @binding(0)
var<storage,read_write> metadata: Metadata;

@group(2) @binding(0)
var<storage> offsets: array<u32>,


const LIST_CHUNK_SIZE: u32 = 4u;

struct Metadata {
    heap_top: atomic<u32>,
}

struct LinkedList {
    data: array<u32,4>,
    len: u32,
    next: u32,
}

struct Position {
    p: u32,
    Q: LinkedList,
}

struct ConjunctionPosition {
    p: u32,
    Q: LinkedList,
    Qx: LinkedList,
}

struct MutList {
    list: LinkedList,
    last: u32,
}

fn new_list() -> LinkedList {
    return LinkedList(
        array<u32,LIST_CHUNK_SIZE>(),  // All zeros
        0u,
        0u,
    );
}

fn singleton(e: u32) -> LinkedList {
    var data = array<u32,LIST_CHUNK_SIZE>();
    data[0] = e;
    return LinkedList(
        data,
        1u,
        0u,
    );
}

@compute
@workgroup_size(64, 1, 1)
fn conjunction_answers(@builtin(global_invocation_id) g_id: vec3<u32>,
                       @builtin(local_invocation_index) l_idx: u32)
{
    let i = g_id.x;
    if i >= arrayLength(&input) {
        return;
    }

    // var-declaration otherwise pos_in.Q.data can only be indexed by a constant
    var pos_in = input[i];
    let p = pos_in.p;

    let part_idx = i % LIST_CHUNK_SIZE;
    let part = i / LIST_CHUNK_SIZE;

    let out_base = offsets[part];
    let out_max = out_base + pos_in.Q.len;
    var pos_idx = out_base + part_idx;
    if pos_idx < out_max {
        let clause_pos = Position(
            p,
            singleton(pos_in.Q.data[part_idx]),
        );
        output[pos_idx] = clause_pos;
    }
    pos_idx += LIST_CHUNK_SIZE;
    var next = pos_in.Q.next;

    while pos_idx < out_max {
        let clause_pos = Position(
            p,
            singleton(heap_in[next].Q.data[part_idx]),
        );
        output[pos_idx] = clause_pos;
        pos_idx += LIST_CHUNK_SIZE;
        next = heap_in[next].next;
    }
}
