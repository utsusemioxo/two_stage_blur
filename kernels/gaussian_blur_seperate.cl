#define CHANNEL_NUM 3

// generate temp result
__kernel void gaussian_blur_rows(
  __global uchar* input,
  __global uchar* temp,
  __constant float* kernel1d,
  int width,
  int height,
  int pitch, 
  int k_w
) {
  int x = get_global_id(0);
  int y = get_global_id(1); 

  if (x >= width || y >= height) {
    return;
  }

  int half_k_w = k_w / 2;

  for (int c = 0; c < CHANNEL_NUM; ++c) {
    float sum = 0.0f;
    for (int kx = 0; kx < k_w; kx++) {
      // calculate row covolution
      int ix = clamp(x + kx - half_k_w, 0, width - 1);
      int iy = y;

      int in_idx = iy * pitch + ix * CHANNEL_NUM + c;
      uchar pixel = input[in_idx];
      float coeff = kernel1d[kx];
      sum += (float)pixel * coeff;
    }

    int out_idx = y * pitch + x * CHANNEL_NUM + c;
    float result = clamp(sum, 0.0f, 255.0f);
    temp[out_idx] = (uchar)result;
  }
}

__kernel void gaussian_blur_cols(
  __global uchar* temp,
  __global uchar* output,
  __constant float* kernel1d,
  int width,
  int height,
  int pitch,
  int k_h
) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x>= width || y >= height) {
    return;
  }

  int half_k_h = k_h / 2;

  for (int c = 0; c < CHANNEL_NUM; ++c) {
    float sum = 0.0f;
    for (int ky = 0; ky < k_h; ky++) {
      // calculate column covolution
      int ix = x;
      int iy = clamp(y + ky - half_k_h, 0, height - 1);

      int in_idx = iy * pitch + ix * CHANNEL_NUM + c;
      uchar pixel = temp[in_idx];
      float coeff = kernel1d[ky];
      sum += (float)pixel * coeff;
    }

    int out_idx = y * pitch + x * CHANNEL_NUM + c;
    float result = clamp(sum, 0.0f, 255.0f);
    output[out_idx] = (uchar)result;

  }
}