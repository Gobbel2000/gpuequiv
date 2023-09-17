@group(0) @binding(0)
var<storage> energies: array<u32>;

@group(0) @binding(1)
var<storage, read_write> minimal: array<u32>;

const n_threads = 64;

var<workgroup> minimal_temp: array<u32,n_threads>;

const n_energies = 6;
const max_mask = 0x3 << n_energies - 1;

fn less(a: u32, b: u32) -> bool {
    if a == b {
        return false;
    }
    for (var mask: u32 = 0x3; mask <= max_mask; mask << 1) {
        if a & mask > b & mask {
            return false;
        }
    }
    return true;
}

@compute
@workgroup_size(n_threads, 1, 1)
fn minima(@builtin(global_invocation_id) g_id: vec3<u32>,
          @builtin(local_invocation_index) l_idx: u32) {
    let e = energies[g_id.x];
    var minimal = 1u;
    for (var j: u32 = 0; j < arrayLength(&energies); j++) {
        if less(energies[j], energies[i]) {
            minimal = 0u;
            break;
        }
    }
    minimal_temp[l_idx] = minimal;

    workgroupBarrier();

    // Reduce all 64 flags into 2 bit-packed u32
    // Valid results only end up in the first elements of both 32-blocks.
    // The other threads just do the same because it is faster than adding conditions.
    // This is also a case of loop-unrolling.
    minimal_temp[l_idx] |= minimal_temp[l_idx + 16];
    minimal_temp[l_idx] |= minimal_temp[l_idx + 8];
    minimal_temp[l_idx] |= minimal_temp[l_idx + 4];
    minimal_temp[l_idx] |= minimal_temp[l_idx + 2];
    minimal_temp[l_idx] |= minimal_temp[l_idx + 1];

    // The first thread in both 32-blocks writes the packed u32 out
    if !(l_idx & 0x1f) { // last 5 bits are 0, l_idx % 32 == 0
        minimal[g_id.x >> 5] = minimal_temp[l_idx];
    }
}
