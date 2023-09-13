@group(0) @binding(0)
var<storage> adj_matrix: array<u32>;

@group(0) @binding(1)
var<storage> attacker_pos: array<u32>;

@group(0) @binding(2)
var<storage, read_write> visit: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output: array<u32>;

fn edge(v_from: u32, to: u32) -> bool {
    let n = arrayLength(&visit);
    return adj_matrix[v_from * n + to] != 0u; 
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&visit);
    if i >= n || visit[i] == 0u {
        return;
    }
    // Reset visit array
    visit[i] = 0u;

    var new_steps: u32;
    if attacker_pos[i] != 0u {
        // Pick minimal step count from next nodes
        new_steps = output[i];
        for (var v: u32 = 0u; v < n; v++) {
            if edge(i, v) && output[v] + 1u < new_steps {
                new_steps = output[v] + 1u;
            }
        }
    } else {
        // Pick maximal step count from next nodes
        new_steps = 0u;
        for (var v: u32 = 0u; v < n; v++) {
            if edge(i, v) && output[v] >= new_steps {
                new_steps = output[v] + 1u;
            }
        }
    }

    if new_steps < output[i] {
        output[i] = new_steps;
        // Visit previous nodes next
        for (var v: u32 = 0u; v < n; v++) {
            if edge(v, i) {
                visit[v] = 1u;
            }
        }
    }
}
