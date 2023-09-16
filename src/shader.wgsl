@group(0) @binding(0)
var<storage> visit: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;


@group(1) @binding(0)
var<storage> graph_c: array<u32>;

@group(1) @binding(1)
var<storage> graph_r: array<u32>;


// Process an attacker position
@compute
@workgroup_size(64, 1, 1)
fn attack(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&visit);
    if i >= n {
        return;
    }
    let v = visit[i];

    // Pick minimal step count from next nodes
    var new_steps = output[v];
    for (var j: u32 = graph_r[v]; j < graph_r[v + 1u]; j++) {
        let w = graph_c[j];
        if output[w] + 1u < new_steps {
            new_steps = output[w] + 1u;
        }
    }

    if new_steps < output[v] {
        output[v] = new_steps;
    }
}

// Process a defense position
@compute
@workgroup_size(64, 1, 1)
fn defend(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&visit);
    if i >= n {
        return;
    }
    let v = visit[i];

    // Pick maximal step count from next nodes
    var new_steps = 0u;
    for (var j: u32 = graph_r[v]; j < graph_r[v + 1u]; j++) {
        let w = graph_c[j];
        if output[w] >= new_steps {
            new_steps = output[w] + 1u;
        }
    }

    if new_steps < output[v] {
        output[v] = new_steps;
    }
}
