// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include <cstdint>

extern "C" int vllm_ppu_gate_silu_launch(const void* x, const void* weight,
                                         void* output, void* stream_handle);

namespace {

constexpr int64_t kM = 1;
constexpr int64_t kK = 2048;
constexpr int64_t kN = 12288;
constexpr int64_t kD = 6144;

}  // namespace

torch::Tensor ppu_gate_silu(const torch::Tensor& x,
                            const torch::Tensor& weight) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(),
              "_C::ppu_gate_silu requires PPU/CUDA tensors");
  TORCH_CHECK(x.device() == weight.device(),
              "x and weight must share a device");
  TORCH_CHECK(
      x.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf,
      "_C::ppu_gate_silu requires FP16 x and weight");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == kM && x.size(1) == kK,
              "x must have shape [1, 2048], got ", x.sizes());
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kN && weight.size(1) == kK,
              "weight must have shape [12288, 2048], got ", weight.sizes());
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
              "x and weight must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
              "x and weight must be 16-byte aligned");

  c10::cuda::CUDAGuard guard(x.device());
  auto output = at::empty({kM, kD}, x.options());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
              "output is not 16-byte aligned");

  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const int status = vllm_ppu_gate_silu_launch(
      x.data_ptr(), weight.data_ptr(), output.mutable_data_ptr(),
      reinterpret_cast<void*>(stream.stream()));
  TORCH_CHECK(status == 0, "HGGCC gate_silu launch failed with error ", status);
  return output;
}
