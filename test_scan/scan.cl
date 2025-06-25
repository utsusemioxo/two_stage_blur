__kernel void scan(
    __global int* data,
    __global int* tile_sums,
    __local int* temp,
    const int N
){ 
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group = get_group_id(0);
    int lsize = get_local_size(0);

    if (gid >= N) return;

    temp[lid] = data[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // up sweep (reduce)
    for (int offset = 1; offset < lsize; offset <<=1) {
        int index = (lid + 1) * offset * 2 - 1;
        if (index < lsize) {
            temp[index] += temp[index - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // set last element to 0 for exclusive scan
    if (lid == lsize - 1) {
        if (tile_sums != NULL) {
            tile_sums[group] = temp[lid];
        }
        temp[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // down sweep
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