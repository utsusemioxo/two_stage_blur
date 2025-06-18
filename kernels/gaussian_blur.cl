#define KERNEL_RADIUS 15
#define ROWS_BLOCKDIM_X 4
#define ROWS_BLOCKDIM_Y 4
#define CHANNEL_NUM 3

__kernel void gaussian_blur(
  __global uchar* input,
  __global uchar* output,
  __global const float* gaussian_kernel,
  int width,
  int height,
  int pitch,
  int k_w,
  int k_h
) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x >= width || y >= height) {
    return;
  }

  int half_k_w = k_w / 2;
  int half_k_h = k_h / 2;

  for (int c = 0; c < CHANNEL_NUM; ++c) {
    float sum = 0.0f;

    for (int ky = 0; ky < k_h; ++ky) {
      for (int kx = 0; kx < k_w; kx++) {
        int ix = x + kx - half_k_w;
        int iy = y + ky - half_k_h;

        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
          int in_idx = iy * pitch + ix * 3 + c;
          uchar pixel = input[in_idx];
          float coeff = gaussian_kernel[ky * k_w + kx];
          sum += (float)pixel * coeff;
        }
      }
    }

    int out_idx = y * pitch + x * 3 + c;
    float result = clamp(sum, 0.0f, 255.0f);
    output[out_idx] = (uchar)result;
  }

}