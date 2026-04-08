#include <torch/extension.h>

torch::Tensor fused_causal_masked_softmax_cpu(torch::Tensor scores) {
  TORCH_CHECK(scores.device().is_cpu(), "scores must be on CPU");
  TORCH_CHECK(scores.dim() == 4, "scores must have shape [B, H, Tq, Tk]");

  auto t_q = scores.size(2);
  auto t_k = scores.size(3);
  auto mask = torch::ones({t_q, t_k}, torch::TensorOptions().dtype(torch::kBool).device(scores.device())).tril();
  auto masked = scores.masked_fill(mask.logical_not().unsqueeze(0).unsqueeze(0), -1e9);
  return torch::softmax(masked, -1);
}

torch::Tensor kv_cache_append_cpu(torch::Tensor cache, torch::Tensor new_values) {
  TORCH_CHECK(cache.device().is_cpu(), "cache must be on CPU");
  TORCH_CHECK(new_values.device().is_cpu(), "new_values must be on CPU");
  TORCH_CHECK(cache.dim() == 4 && new_values.dim() == 4, "cache and new_values must have shape [B, H, T, D]");
  return torch::cat({cache, new_values}, 2);
}

torch::Tensor fused_causal_masked_softmax_cuda(torch::Tensor scores);
torch::Tensor kv_cache_append_cuda(torch::Tensor cache, torch::Tensor new_values);

torch::Tensor fused_causal_masked_softmax(torch::Tensor scores) {
  if (scores.is_cuda()) {
    return fused_causal_masked_softmax_cuda(scores);
  }
  return fused_causal_masked_softmax_cpu(scores);
}

torch::Tensor kv_cache_append(torch::Tensor cache, torch::Tensor new_values) {
  if (cache.is_cuda()) {
    return kv_cache_append_cuda(cache, new_values);
  }
  return kv_cache_append_cpu(cache, new_values);
}
