// SPDX-License-Identifier: Apache-2.0
// MXFP4 CUTLASS Block-Scaled GEMM Entry Point

#include <torch/all.h>

#if defined ENABLE_MXFP4_SM120 && ENABLE_MXFP4_SM120
void cutlass_scaled_mxfp4_mm_sm120a(torch::Tensor& D, torch::Tensor const& A,
                                     torch::Tensor const& B,
                                     torch::Tensor const& A_sf,
                                     torch::Tensor const& B_sf);
#endif

void cutlass_scaled_mxfp4_mm(torch::Tensor& D, torch::Tensor const& A,
                              torch::Tensor const& B,
                              torch::Tensor const& A_sf,
                              torch::Tensor const& B_sf) {
#if defined ENABLE_MXFP4_SM120 && ENABLE_MXFP4_SM120
  return cutlass_scaled_mxfp4_mm_sm120a(D, A, B, A_sf, B_sf);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "No compiled mxfp4 mm kernel, vLLM should "
                              "be compiled using CUDA 12.8 and target "
                              "compute capability 120.");
}

bool cutlass_scaled_mm_supports_mxfp4(int64_t cuda_device_capability) {
  int runtimeVersion;
  cudaRuntimeGetVersion(&runtimeVersion);
  return cuda_device_capability == 120 && runtimeVersion >= 12080;
}
