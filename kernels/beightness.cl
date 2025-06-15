// brightness.cl
__kernel void brighten(__global uchar* image, int width, int height, uchar value) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * width + x;
    if (x < width && y < height) {
        uchar pixel = image[idx];
        uchar new_pixel = clamp((int)pixel + value, 0, 255);
        image[idx] = new_pixel;
    }
}
