#include <torch/extension.h>

torch::Tensor fused_causal_masked_softmax(torch::Tensor scores);
torch::Tensor kv_cache_append(torch::Tensor cache, torch::Tensor new_values);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_causal_masked_softmax", &fused_causal_masked_softmax, "Fused causal masked softmax");
  m.def("kv_cache_append", &kv_cache_append, "Append new key/value chunk to cache");
}
