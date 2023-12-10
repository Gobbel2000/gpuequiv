@group(0) @binding(0)
var<storage> input: array<ConjunctionPosition>;

@group(0) @binding(1)
var<storage, read_write> output: array<Position>;

// Dynamically allocated memory area for storing arbitrarily sized sets of
// processes alongside each position in the form of linked lists.
@group(0) @binding(2)
var<storage> heap_in: array<LinkedList>;

@group(0) @binding(3)
var<storage, read_write> heap: array<LinkedList>;


@group(1) @binding(0)
var<storage,read_write> metadata: Metadata;



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
        array<u32,4>(),  // All zeros
        0u,
        0u,
    );
}

fn copy_list(src: LinkedList) -> LinkedList {
    if src.len <= 4u {
        return src;
    }

    var new = src;

    // Pointer to current element in old list
    var src_next = src.next;

    // Pointer to previous element in new linked list
    var prev = atomicAdd(&metadata.heap_top, 1u);
    if prev > arrayLength(&heap) {
        return;
    }
    var element = heap_in[src_next];
    src_next = element.next;
    new.next = prev;
    heap[prev] = element;
    while element.len > 4u {
        let pointer = atomicAdd(&metadata.heap_top, 1u);
        if pointer > arrayLength(&heap) {
            return;
        }
        element = heap_in[src_next];
        src_next = element.next;
        heap[prev].next = pointer;
        heap[pointer] = element;
        prev = pointer;
    }
}


@compute
@workgroup_size(64, 1, 1)
fn process_challenges(@builtin(global_invocation_id) g_id: vec3<u32>,
                      @builtin(local_invocation_index) l_idx: u32)
{
    let i = g_id.x;
    if i >= arrayLength(&input) {
        return;
    }

    // var-declaration otherwise pos_in.Q.data can only be indexed by a constant
    var pos_in = input[i];
    let p = pos_in.p;

    // Copy Q* of input position into new position
    let pos_Qx = Position(
        p,
        copy_list(pos_in.Qx),
    );

    output[i] = pos_Qx;

    let q_size = pos_in.Q.len;
}
