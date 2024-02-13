alias Energy = array<u32, $ENERGY_SIZE>;

@group(0) @binding(0)
var<storage, read_write> energies: array<Energy>;

@group(0) @binding(1)
var<storage> node_offsets: array<NodeOffset>;

@group(0) @binding(2)
var<storage> successor_offsets: array<u32>;

@group(1) @binding(0)
// Final, minimized energies. Each node gets a certain amount of preallocated
// memory in this array. If that is not enough, it will be run again with more.
// The memory size in elements each node gets is always a multiple of the
// workgroup size, which is 64.
var<storage, read_write> suprema: array<Energy>;

@group(1) @binding(1)
// Returns the final size of each suprema array.
// Negative values indicate that a node had insufficient memory in its suprema
// array.
// One entry per node.
var<storage, read_write> status: array<i32>;


struct NodeOffset {
    node: u32,
    successor_offsets_idx: u32,
    energy_offset: u32,
    // Start of allocated region in suprema array
    sup_offset: u32,
}

// Temporary buffer to hold the output flags as u32. This is later condensed
// into bit-packed u32's in the minima storage buffer.
var<workgroup> minima_buf: array<u32,64u>;
var<workgroup> minima_total: u32;

fn less_eq(a: Energy, b: Energy) -> bool {
    $IMPL_LESS_EQ;
}

fn energy_eq(a: Energy, b: Energy) -> bool {
    $IMPL_EQ;
}

fn unpack_energy(e: Energy) -> array<u32, $ENERGY_ELEMENTS> {
    return $UNPACK_ENERGY;
}

// Compute a prefix sum over the minima_buf array.
// As a result, element i will contain the sum of elements
// minima_buf[0] + ... + minima_buf[i - 1].
// In this context that is the number of empty spots (filtered out
// energies) to the left of i.
//
// Only half of all threads (0..32 in each workgroup) contribute in the
// prefix sum, because the lowest operation is comparing two elements.
//
// Additionally, the workgroup variable `minima_total` is set to the full sum
// of the array.
//
// This part is heavily based on code implementing the Prefix Sum
// algorithm in WebGPU by Will Usher, published under the MIT License at:
// https://github.com/Twinklebear/webgpu-experiments/blob/a5b8dfc4dbba651c814c235e1756f17fc062422a/shaders/prefix_sum.comp
//
// The algorithm as implemented is described in
// "Parrallel Prefix Sum (Scan) with CUDA" 2007 by Mark Harris:
// https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
fn prefix_sum(l_idx: u32) {
    // Up-sweep (reduce) phase
    var offset = 1u;
    for (var d = 32u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if l_idx < d {
            let ai = offset * (2u * l_idx + 1u) - 1u;
            let bi = offset * (2u * l_idx + 2u) - 1u;
            minima_buf[bi] += minima_buf[ai];
        }
        offset <<= 1u;
    }

    // The last element now contains the total sum. We set it to 0.
    if l_idx == 0u {
        minima_total = minima_buf[63u];
        minima_buf[63u] = 0u;
    }

    // Down-sweep phase
    for (var d = 1u; d < 64u; d <<= 1u) {
        offset >>= 1u;
        workgroupBarrier();
        if l_idx < d {
            let ai = offset * (2u * l_idx + 1u) - 1u;
            let bi = offset * (2u * l_idx + 2u) - 1u;
            let swap = minima_buf[ai];
            minima_buf[ai] = minima_buf[bi];
            minima_buf[bi] += swap;
        }
    }
    workgroupBarrier();
}

// Minimizes the energies suprema[sbase]..suprema[sbase + size] by removing
// non-minimal elements and shifting the others as far as possible to the left
// to fill any gaps.
//
// extra_shift is the number of energies from the previous iteration. They are
// not needed anymore and the new energies will be shifted on top of them.
//
// Returns the final number of minimal energies.
fn minimize(sbase: u32, extra_shift: u32, size: u32, l_idx: u32) -> u32 {
    // Initialize number of filtered out elements with extra_shift, so that
    // minimal energies get shifted down to sbase.
    var total_filtered = extra_shift;
    var n_minimized = 0u;
    // Index of first considered energy
    let nbase = sbase + extra_shift;
    for (var chunk = 0u; chunk < size; chunk += 64u) {
        let i = chunk + l_idx;
        let extra_i = i + extra_shift;
        let energy = suprema[nbase + i];
        // Minimize energies
        var filter_out = 0u;
        if i < size {
            // Compare with already minimized energies that have been shifted down to `sbase`
            for (var j = 0u; j < n_minimized; j += 1u) {
                let e2 = suprema[sbase + j];
                // Skip reflexive comparisons,
                // When energies are equal, keep only those with lower index
                let eq = energy_eq(e2, energy);
                if j != extra_i && ((eq && extra_i > j) ||
                              (!eq && less_eq(e2, energy))) {
                    // Mark to be filtered out
                    filter_out = 1u;
                    break;
                }
            }
            if filter_out == 0u {
                // If not yet marked for removal, check all following energies,
                // that have not yet been minimized.
                for (var j = chunk; j < size; j += 1u) {
                    let e2 = suprema[nbase + j];
                    // Skip reflexive comparisons,
                    // When energies are equal, keep only those with lower index
                    let eq = energy_eq(e2, energy);
                    if j != i && ((eq && i > j) ||
                                  (!eq && less_eq(e2, energy))) {
                        // Mark to be filtered out
                        filter_out = 1u;
                        break;
                    }
                }
            }
        }
        minima_buf[l_idx] = filter_out;

        prefix_sum(l_idx);
        n_minimized += min(size - chunk, 64u) - minima_total;

        // Shift energies left to fill any gaps left by non-minimal energies
        storageBarrier();
        if filter_out == 0u && i < size { // Only shift minimal energies
            // Shift by number of filtered elements in this chunk +
            // total previously filtered energies.
            let shift = minima_buf[l_idx] + total_filtered;
            suprema[nbase + i - shift] = energy;
        }
        storageBarrier();
        total_filtered += minima_total;
    }
    return n_minimized;
}

@compute
@workgroup_size(64, 1, 1)
fn intersection(@builtin(workgroup_id) wg_id: vec3<u32>,
                @builtin(local_invocation_index) l_idx: u32)
{
    let node = node_offsets[wg_id.x];
    let next = node_offsets[wg_id.x + 1u];
    // This node has no successors, abort
    if next.successor_offsets_idx == node.successor_offsets_idx {
        return;
    }
    // This workgroup's base pointer for the output array
    let sbase = node.sup_offset;
    // Amount of memory available for this workgroup to write the suprema for
    // this node. If it turns out to be insufficient, the shader will be
    // invoked again with more allocated memory.
    let available_mem = node_offsets[wg_id.x + 1u].sup_offset - sbase;

    // Copy (updated) energies of first successor
    var slen = successor_offsets[node.successor_offsets_idx + 1u] - node.energy_offset;
    if slen > available_mem {
        // Not enough memory, abort.
        status[wg_id.x] = - i32(slen);
        return;
    }
    for (var chunk = 0u; chunk < slen; chunk += 64u) {
        let i = chunk + l_idx;
        if i < slen {
            suprema[sbase + i] = energies[node.energy_offset + i];
        }
    }
    storageBarrier();

    for (var suc = node.successor_offsets_idx + 1u;
         suc < next.successor_offsets_idx;
         suc++)
    {
        // Number of energies for this next successor
        let suc_width = successor_offsets[suc + 1u] - successor_offsets[suc];
        // We will combine each previous energy with each energy of this successor
        let comb_size = slen * suc_width;
        // Where to write the next chunk of combinations. This depends on the
        // number of minimal energies left after each chunk of combinations.
        var n_minimized = 0u;
        // Combine with energies from next successor
        for (var chunk = 0u; chunk < comb_size; chunk += 64u) {
            let required_mem = slen + n_minimized + min(comb_size - chunk, 64u);
            if required_mem > available_mem {
                // Not enough memory, abort.
                status[wg_id.x] = - i32(required_mem);
                return;
            }
            let i = chunk + l_idx;
            if i < comb_size {
                let prev_pick = i % slen;
                let next_pick = i / slen;
                // Previously added energy. This array will get mutated in place
                var supremum = unpack_energy(suprema[sbase + prev_pick]);
                let e = energies[successor_offsets[suc] + next_pick];
                let energy = unpack_energy(e);

                //supremum[0u] = max(supremum[0u], energy[0u]);
                //supremum[1u] = max(supremum[1u], energy[1u]);
                //...
                $MAX_SUPREMUM

                // Write combinations behind previous minimized energies
                suprema[sbase + slen + n_minimized + l_idx] = $PACK_SUPREMUM;
            }
            // Ensure that all threads have written their supremum.
            storageBarrier();

            // Minimize each chunk
            if chunk + 64u >= comb_size {
                // Last chunk, also shift over previous energies
                let size = n_minimized + comb_size - chunk;
                slen = minimize(sbase, slen, size, l_idx);
            } else {
                n_minimized = minimize(sbase + slen, 0u, n_minimized + 64u, l_idx);
            }
        }
    }
    if l_idx == 0u {
        status[wg_id.x] = i32(slen);
    }
}
