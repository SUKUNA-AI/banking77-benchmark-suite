#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void fused_causal_masked_softmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch,
    int heads,
    int t_q,
    int t_k) {
  const int row = blockIdx.x;
  const int row_count = batch * heads * t_q;
  if (row >= row_count) {
    return;
  }

  const int q_idx = row % t_q;
  const int head_idx = (row / t_q) % heads;
  const int batch_idx = row / (t_q * heads);
  const int base = ((batch_idx * heads + head_idx) * t_q + q_idx) * t_k;

  __shared__ float shared_max_buffer[256];
  __shared__ float shared_sum_buffer[256];
  float max_val = -1e30f;
  for (int k = threadIdx.x; k < t_k; k += blockDim.x) {
    float value = (k <= q_idx) ? static_cast<float>(input[base + k]) : -1e30f;
    if (value > max_val) {
      max_val = value;
    }
  }
  shared_max_buffer[threadIdx.x] = max_val;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max_buffer[threadIdx.x] = max(shared_max_buffer[threadIdx.x], shared_max_buffer[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float shared_max = shared_max_buffer[0];

  float local_sum = 0.0f;
  for (int k = threadIdx.x; k < t_k; k += blockDim.x) {
    float weight = 0.0f;
    if (k <= q_idx) {
      weight = expf(static_cast<float>(input[base + k]) - shared_max);
    }
    output[base + k] = static_cast<scalar_t>(weight);
    local_sum += weight;
  }
  shared_sum_buffer[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum_buffer[threadIdx.x] += shared_sum_buffer[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float denom = max(shared_sum_buffer[0], 1e-8f);
  for (int k = threadIdx.x; k < t_k; k += blockDim.x) {
    output[base + k] = static_cast<scalar_t>(static_cast<float>(output[base + k]) / denom);
  }
}

template <typename scalar_t>
__global__ void kv_cache_append_kernel(
    const scalar_t* __restrict__ cache,
    const scalar_t* __restrict__ update,
    scalar_t* __restrict__ output,
    int batch,
    int heads,
    int cache_t,
    int update_t,
    int dim) {
  const int total_t = cache_t + update_t;
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * heads * total_t * dim;
  if (flat >= total) {
    return;
  }

  const int d = flat % dim;
  const int t = (flat / dim) % total_t;
  const int h = (flat / (dim * total_t)) % heads;
  const int b = flat / (dim * total_t * heads);

  const int out_index = (((b * heads + h) * total_t + t) * dim) + d;
  if (t < cache_t) {
    const int cache_index = (((b * heads + h) * cache_t + t) * dim) + d;
    output[out_index] = cache[cache_index];
  } else {
    const int update_index = (((b * heads + h) * update_t + (t - cache_t)) * dim) + d;
    output[out_index] = update[update_index];
  }
}

}  // namespace

torch::Tensor fused_causal_masked_softmax_cuda(torch::Tensor scores) {
  TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
  TORCH_CHECK(scores.dim() == 4, "scores must have shape [B, H, Tq, Tk]");
  auto output = torch::zeros_like(scores);
  const int batch = static_cast<int>(scores.size(0));
  const int heads = static_cast<int>(scores.size(1));
  const int t_q = static_cast<int>(scores.size(2));
  const int t_k = static_cast<int>(scores.size(3));
  const int rows = batch * heads * t_q;
  const int threads = 256;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scores.scalar_type(),
      "fused_causal_masked_softmax_cuda",
      [&] {
        fused_causal_masked_softmax_kernel<scalar_t><<<rows, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            scores.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch,
            heads,
            t_q,
            t_k);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor kv_cache_append_cuda(torch::Tensor cache, torch::Tensor new_values) {
  TORCH_CHECK(cache.is_cuda() && new_values.is_cuda(), "cache and new_values must be CUDA");
  TORCH_CHECK(cache.dim() == 4 && new_values.dim() == 4, "cache and new_values must have shape [B, H, T, D]");
  TORCH_CHECK(cache.size(0) == new_values.size(0), "batch must match");
  TORCH_CHECK(cache.size(1) == new_values.size(1), "heads must match");
  TORCH_CHECK(cache.size(3) == new_values.size(3), "head dim must match");

  auto total_t = cache.size(2) + new_values.size(2);
  auto output = torch::empty({cache.size(0), cache.size(1), total_t, cache.size(3)}, cache.options());
  const int total = static_cast<int>(cache.size(0) * cache.size(1) * total_t * cache.size(3));
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      cache.scalar_type(),
      "kv_cache_append_cuda",
      [&] {
        kv_cache_append_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            cache.data_ptr<scalar_t>(),
            new_values.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<int>(cache.size(0)),
            static_cast<int>(cache.size(1)),
            static_cast<int>(cache.size(2)),
            static_cast<int>(new_values.size(2)),
            static_cast<int>(cache.size(3)));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
