@group(0) @binding(0)
var<storage> input: array<Position>;

@group(0) @binding(1)
var<storage, read_write> output: array<ConjunctionPosition>;

// Dynamically allocated memory area for storing arbitrarily sized sets of
// processes alongside each position in the form of linked lists.
@group(0) @binding(2)
var<storage> heap_in: array<LinkedList>;

@group(0) @binding(3)
var<storage, read_write> heap: array<LinkedList>;


@group(1) @binding(0)
var<storage,read_write> metadata: Metadata;


// Input labeled transition system in CSR format
@group(2) @binding(0)
var<storage> graph_column_indices: array<u32>;

@group(2) @binding(1)
var<storage> graph_row_offsets: array<u32>;

@group(2) @binding(2)
var<storage> graph_labels: array<u32>;



struct Metadata {
    heap_top: atomic<u32>,
    heap_oom: u32,
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

fn list_append(l: ptr<private,MutList>, element: u32) {
    var last = (*l).last;
    if (*l).list.len < 4u {
        // Fits in first part
        (*l).list.data[(*l).list.len] = element;
        (*l).list.len += 1u;
    } else if (*l).list.len > 4u && heap[last].len < 4u {
        // Fits in last allocated part
        heap[last].data[heap[last].len] = element;
        heap[last].len += 1u;
    } else {
        // Requires allocating element part
        let next_part = LinkedList(
            array<u32,4>(element, 0u, 0u, 0u),
            1u,
            0u,
        );
        let pointer = atomicAdd(&metadata.heap_top, 1u);
        if pointer > arrayLength(&heap) {
            metadata.heap_oom = 1u;
            return;
        }
        heap[pointer] = next_part;

        if (*l).list.len == 4u {
            (*l).list.next = pointer;
            (*l).list.len += 1u;
        } else {
            heap[last].next = pointer;
            heap[last].len += 1u;
        }
        last = pointer;
    }
    (*l).last = last;
}

// This functions requires all adjacency lists to be sorted by their label in
// ascending order.
fn compare_enabled(p: u32, q: u32) -> vec2<bool> {
    // Store two boolean flags in a vector:
    //
    // First element (subset):     I(q) ⊆ I(p)
    // Second element (superset):  I(q) ⊇ I(p)
    //
    // I(u) represents the set of enabled actions of node u in the input LTS.
    var out = vec2<bool>(true);

    if p == q {
        return out;
    }

    var p_ptr = graph_row_offsets[p];
    let p_max = graph_row_offsets[p + 1u] - 1u;
    var q_ptr = graph_row_offsets[q];
    let q_max = graph_row_offsets[q + 1u] - 1u;

    loop {
        if !any(out) {
            // Already incomparable
            break;
        }

        if p_ptr == p_max && q_ptr == q_max {
            // Both lists exhausted
            break;
        }
        if p_ptr == p_max {
            // Exhausted p, q has additional members, so is not a subset of p
            out.x = false;
            break;
        }
        if q_ptr == q_max {
            // Exhausted q, p has additional members. q is not a superset of p
            out.y = false;
            break;
        }

        let p_act = graph_labels[p_ptr];
        let q_act = graph_labels[q_ptr];

        var advance_p = false;
        var advance_q = false;

        if p_act == q_act {
            // Both p and q have this action, continue
            advance_p = true;
            advance_q = true;
        } else if p_act < q_act {
            // p has an action that q doesn't have
            out.y = false;
            advance_p = true;
        } else if q_act < p_act {
            // q has an action that p doesn't have
            out.x = false;
            advance_q = true;
        }

        if advance_p {
            // Skip ahead until next distinct action, or end of list, but
            // always at least by one.
            p_ptr += 1u;
            while p_ptr < p_max && graph_labels[p_ptr] == p_act {
                p_ptr += 1u;
            }
        }
        if advance_q {
            q_ptr += 1u;
            while q_ptr < q_max && graph_labels[q_ptr] == q_act {
                q_ptr += 1u;
            }
        }
    }
    return out;
}

var<private> Q_subset: MutList;
var<private> Qx_subset: MutList;
var<private> Q_superset: MutList;
var<private> Qx_superset: MutList;
var<private> Q_equal: MutList;
var<private> Qx_equal: MutList;

@compute
@workgroup_size(64, 1, 1)
fn process_challenges(@builtin(global_invocation_id) g_id: vec3<u32>,
                      @builtin(local_invocation_index) l_idx: u32)
{
    let i = g_id.x;
    if i >= arrayLength(&input) {
        return;
    }

    // var-declaration otherwise atk_pos.Q.data can only be indexed by a constant
    var atk_pos = input[i];
    let p = atk_pos.p;

    // Copy atk_pos to new conjuction position using empty Qx
    var pos_empty = ConjunctionPosition(
        p,
        atk_pos.Q,
        new_list(),
    );
    // Pointer to current element in old list
    var cur = pos_empty.Q.next;
    // Pointer to previous element in new linked list
    var prev = 0u;
    if cur != 0u {
        prev = atomicAdd(&metadata.heap_top, 1u);
        if prev > arrayLength(&heap) {
            metadata.heap_oom = 1u;
            return;
        }
        let element = heap_in[cur];
        cur = element.next;
        pos_empty.Q.next = prev;
        heap[prev] = element;
    }
    while cur != 0u {
        let pointer = atomicAdd(&metadata.heap_top, 1u);
        if pointer > arrayLength(&heap) {
            metadata.heap_oom = 1u;
            return;
        }
        let element = heap_in[cur];
        cur = element.next;
        heap[prev].next = pointer;
        heap[pointer] = element;
        prev = pointer;
    }

    for (var i = 0u; i < min(atk_pos.Q.len, 4u); i++) {
        let q = atk_pos.Q.data[i];
        let cmp = compare_enabled(p, q);
        if cmp.x {
            list_append(&Qx_subset, q);
        } else {
            list_append(&Q_subset, q);
        }
        if cmp.y {
            list_append(&Qx_superset, q);
        } else {
            list_append(&Q_superset, q);
        }
        if all(cmp) {
            list_append(&Qx_equal, q);
        } else {
            list_append(&Q_equal, q);
        }
    }
    if atk_pos.Q.len > 4u {
        var cur = atk_pos.Q.next;
        for (var i = 0u; i < min(heap_in[cur].len, 4u); i++) {
            let q = heap_in[cur].data[i];
            let cmp = compare_enabled(p, q);
            if cmp.x {
                list_append(&Qx_subset, q);
            } else {
                list_append(&Q_subset, q);
            }
            if cmp.y {
                list_append(&Qx_superset, q);
            } else {
                list_append(&Q_superset, q);
            }
            if all(cmp) {
                list_append(&Qx_equal, q);
            } else {
                list_append(&Q_equal, q);
            }
        }
    }

    output[i * 4u] = pos_empty;
    output[i * 4u + 1u] = ConjunctionPosition(
        p,
        Q_subset.list,
        Qx_subset.list,
    );
    output[i * 4u + 2u] = ConjunctionPosition(
        p,
        Q_superset.list,
        Qx_superset.list,
    );
    output[i * 4u + 3u] = ConjunctionPosition(
        p,
        Q_equal.list,
        Qx_equal.list,
    );

}
