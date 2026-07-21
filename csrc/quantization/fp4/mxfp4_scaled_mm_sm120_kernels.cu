// SPDX-License-Identifier: Apache-2.0
// MXFP4 CUTLASS Block-Scaled GEMM Kernel for SM120 (GeForce Blackwell)
//
// Key differences from NVFP4 SM120:
//   - mx_float4_t instead of nv_float4_t
//   - E8M0 (float_ue8m0_t) scale factors instead of E4M3 (float_ue4m3_t)
//   - Group size 32 (Sm1xxBlockScaledConfig<32>) instead of 16
//   - No alpha parameter (implicit alpha = 1.0)

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass_extensions/common.hpp"

#include "cutlass/cutlass.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "core/math.hpp"

using namespace cute;

#define CHECK_TYPE(x, st, m) \
  TORCH_CHECK(x.scalar_type() == st, ": Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) \
  TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

constexpr auto MXFP4_FLOAT4_E2M1X2 = at::ScalarType::Byte;
// E8M0 scale factors stored as uint8 (Byte)
constexpr auto MXFP4_SF_DTYPE = at::ScalarType::Byte;

struct sm120_mxfp4_config_M256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

struct sm120_mxfp4_config_default {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

template <typename Config, typename OutType>
struct Mxfp4GemmSm120 {
  // A matrix configuration — mx_float4_t instead of nv_float4_t
  using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  // B matrix configuration
  using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  // Kernel functional config
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
typename Gemm::Arguments args_from_options(at::Tensor& D, at::Tensor const& A,
                                           at::Tensor const& B,
                                           at::Tensor const& A_sf,
                                           at::Tensor const& B_sf,
                                           int M, int N, int K) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  // E8M0 scale factors for MXFP4
  using ElementSFA = cutlass::float_ue8m0_t;
  using ElementSFB = cutlass::float_ue8m0_t;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {{},  // epilogue.thread — no alpha for MXFP4 (default 1.0)
       static_cast<ElementD const*>(D.data_ptr()),
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  // No fusion_args.alpha_ptr needed for MXFP4

  return arguments;
}

template <typename Gemm>
void runGemm(at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
             at::Tensor const& A_sf, at::Tensor const& B_sf,
             int M, int N, int K, cudaStream_t stream) {
  Gemm gemm;

  auto arguments = args_from_options<Gemm>(D, A, B, A_sf, B_sf, M, N, K);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));

  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

void cutlass_mxfp4_bf16_gemm_dispatch_sm120(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    int m, int n, int k, cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 256) {
    runGemm<Mxfp4GemmSm120<sm120_mxfp4_config_M256,
                            cutlass::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, m, n, k, stream);
  } else {
    runGemm<Mxfp4GemmSm120<sm120_mxfp4_config_default,
                            cutlass::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, m, n, k, stream);
  }
}

void cutlass_mxfp4_f16_gemm_dispatch_sm120(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    int m, int n, int k, cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 256) {
    runGemm<Mxfp4GemmSm120<sm120_mxfp4_config_M256,
                            cutlass::half_t>::Gemm>(
        D, A, B, A_sf, B_sf, m, n, k, stream);
  } else {
    runGemm<Mxfp4GemmSm120<sm120_mxfp4_config_default,
                            cutlass::half_t>::Gemm>(
        D, A, B, A_sf, B_sf, m, n, k, stream);
  }
}

void cutlass_scaled_mxfp4_mm_sm120a(torch::Tensor& D, torch::Tensor const& A,
                                     torch::Tensor const& B,
                                     torch::Tensor const& A_sf,
                                     torch::Tensor const& B_sf) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_INPUT(A, MXFP4_FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, MXFP4_FLOAT4_E2M1X2, "b");

  CHECK_INPUT(A_sf, MXFP4_SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, MXFP4_SF_DTYPE, "scale_b");

  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(A.sizes()[1] == B.sizes()[1],
              "a and b shapes cannot be multiplied (", A.sizes()[0], "x",
              A.sizes()[1], " and ", B.sizes()[0], "x", B.sizes()[1], ")");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment,
              ", but got a shape: (", A.sizes()[0], "x", A.sizes()[1],
              "), k: ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got b shape: (", B.sizes()[0], "x", B.sizes()[1], ").");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  // MXFP4: k / 32 instead of k / 16 (group size 32)
  int rounded_k = round_up(k / 32, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(A_sf.sizes()[1] == B_sf.sizes()[1],
              "scale_a and scale_b shapes cannot be multiplied (",
              A_sf.sizes()[0], "x", A_sf.sizes()[1], " and ", B_sf.sizes()[0],
              "x", B_sf.sizes()[1], ")");
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to a shape (", rounded_m,
              "x", rounded_k, "), but got a shape (", A_sf.sizes()[0], "x",
              A_sf.sizes()[1], ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to a shape (", rounded_n,
              "x", rounded_k, "), but got a shape (", B_sf.sizes()[0], "x",
              B_sf.sizes()[1], ")");

  auto out_dtype = D.dtype();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  if (out_dtype == at::ScalarType::BFloat16) {
    return cutlass_mxfp4_bf16_gemm_dispatch_sm120(D, A, B, A_sf, B_sf, m, n, k,
                                                   stream);
  } else if (out_dtype == at::ScalarType::Half) {
    return cutlass_mxfp4_f16_gemm_dispatch_sm120(D, A, B, A_sf, B_sf, m, n, k,
                                                  stream);
  } else {
    TORCH_CHECK(false, "Unsupported output data type of mxfp4 mm sm120 (",
                out_dtype, ")");
  }
#else
  TORCH_CHECK(false,
              "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
              "a CUTLASS 3.8 source directory to enable support.");
#endif  // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
}
