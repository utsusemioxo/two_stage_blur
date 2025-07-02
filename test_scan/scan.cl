// ------------------------------------------------------------------
// This kernel performs an exclusive prefix sum (also known as scan)
// on a block of data using Blelloch's algorithm (work-efficient scan)
//
// Each work-group operates on a tile of data independently,
// using local memory as a scratch space for summation magic.
// Results are written back to global memory, and optionally,
// the total sum of each tile can be stored in `tile_sums`.
// ------------------------------------------------------------------

__kernel void scan(
    __global int* data,        // Input/output array in global memory
    __global int* tile_sums,   // Optional output of per-tile total sums
    __local int* temp,         // Local scratch space for scan
    const int N                // Total number of elemenets
){ 
    // Identify this thread's global and local position
    int gid = get_global_id(0);    // Absolute position in the global arena
    int lid = get_local_id(0);     // Position inside the current work-group
    int group = get_group_id(0);   // Which work-group this is
    int lsize = get_local_size(0); // The size of this work-group

    // Out-of-bounds check: do not process data beyond the end
    if (gid >= N) return;

    // Phase 1: Load data into the shared local memory
    temp[lid] = data[gid];
    barrier(CLK_LOCAL_MEM_FENCE);  // Wait for all threads to complete load

    // Phase 2: Up-sweep (Reduction)
    // Accmulate values in a binary tree fashion
    // index:   0   1   2   3   4   5   6   7
    // temp:   [a0][a1][a2][a3][a4][a5][a6][a7]
    //
    // offset = 1:
    // [a0][a0+a1][a2][a2+a3][a4][a4+a5][a6][a6+a7]
    // => temp[1] += temp[0]
    // => temp[3] += temp[2]
    // => temp[5] += temp[4]
    // => temp[7] += temp[6]
    //
    // offset = 2:
    // [a0][a0+a1][a2][a0+a1+a2+a3][a4][a4+a5][a6][a4+a5+a6+a7]
    // => temp[3] += temp[1]
    // => temp[7] += temp[5]
    //
    // offset = 4:
    // [a0][a0+a1][a2][a0+a1+a2+a3][a4][a4+a5][a6][a0+a1+a2+a3+a4+a5+a6+a7]
    // => temp[7] += temp[3]
    // Now temp[7] holds the total sum
    for (int offset = 1; offset < lsize; offset <<=1) { // from leaf to root
        int index = (lid + 1) * offset * 2 - 1;
        if (index < lsize) {
            temp[index] += temp[index - offset];
        }

        // Because some thread use other thread's result in formor steps
        // we need to sync after each step, prevent race condition
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Phase 3: Prepare for exclusive scan
    // set last element to 0 for exclusive scan
    if (lid == lsize - 1) {
        if (tile_sums != NULL) {
            // Save total sum of this tile
            tile_sums[group] = temp[lid];
        }
        temp[lid] = 0;
    }

    // Sync before swap in Down-sweep Phase
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 4: Down-sweep (Distribution)
    // Traversing the tree backwards to build exclusive prefix sums
    // Use local size of 7 for example:
    //
    // Initial:
    // T7:
    //   temp[7] = 0
    // step0 res: [a0] [a0+a1] [a2] [a0+a1+a2+a3] [a4] [a4+a5] [a6] [0]
    // 
    // offset = 4, local_thread_index=0, temp_index=7
    // T0:
    //   t = temp[3](a0+a1+a2+a3)
    //   temp[3] = temp[7](0)
    //   temp[7] = t(a0+a1+a2+a3) + temp[7](0)
    // step1 res: [a0] [a0+a1] [a2] [0] [a4] [a4+a5] [a6] [a0+a1+a2+a3]
    //
    // offset = 2, local_thread_index=0,1, temp_index=3,7
    // T0:
    //   t = temp[1](a0+a1)
    //   temp[1] = temp[3](0)
    //   temp[3] = t(a0+a1) + temp[3](0)
    // T1:
    //   t = temp[5](a4+a5)
    //   temp[5] = temp[7](a0+a1+a2+a3)
    //   temp[7] = t(a4+a5) + temp[7](a0+a1+a2+a3)
    // step2 res: [a0] [0] [a2] [a0+a1] [a4] [a0+a1+a2+a3] [a6] [a0+a1+a2+a3+a4+a5]
    //
    // offset = 1, local_thread_index=0,1,2,3, temp_index=1,3,5,7
    // T0:
    //   t = temp[0](a0)
    //   temp[0] = temp[1](0)
    //   temp[1] = t(a0) + temp[1](0) 
    // T1:
    //   t = temp[2](a2)
    //   temp[2] = temp[3](a0+a1)
    //   temp[3] = t(a2) + temp[3](a0+a1)
    // T2:
    //   t = temp[4](a4)
    //   temp[4] = temp[5](a0+a1+a2+a3)
    //   temp[5] = t(a4) + temp[5](a0+a1+a2+a3)
    // T3:
    //   t = temp[6](a6)
    //   temp[6] = temp[7](a0+a1+a2+a3+a4+a5)
    //   temp[7] = t(a6) + temp[7](a0+a1+a2+a3+a4+a5)
    // step3 res: [0] [a0] [a0+a1] [a0+a1+a2] [a0+a1+a2+a3] [a0+a1+a2+a3+a4] [a0+a1+a2+a3+a4+a5] [a0+a1+a2+a3+a4+a5+a6]
    for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
        int index = (lid + 1) * offset * 2 - 1;
        if (index < lsize) {
            int t = temp[index - offset];
            temp[index - offset] = temp[index];
            temp[index] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    data[gid] = temp[lid];
}


__kernel void uniform_add(
    __global int* data,
    __global int* tile_sums,
    int N,
    int TILE_SIZE
) {
    int gid = get_global_id(0);
    int tile_id = gid / TILE_SIZE;
    data[gid] += tile_sums[tile_id];
}