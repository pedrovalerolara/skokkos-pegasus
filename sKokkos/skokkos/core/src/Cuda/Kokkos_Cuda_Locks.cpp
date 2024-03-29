//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Core.hpp>
#ifdef KOKKOS_ENABLE_CUDA
#include <Cuda/Kokkos_Cuda_Locks.hpp>
#include <Cuda/Kokkos_Cuda_Error.hpp>

#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
namespace Kokkos {
namespace Impl {
__device__ __constant__ CudaLockArrays g_device_cuda_lock_arrays = {nullptr, 0};
}
}  // namespace Kokkos
#endif

namespace Kokkos {

namespace {

__global__ void init_lock_array_kernel_atomic() {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < CUDA_SPACE_ATOMIC_MASK + 1) {
    Kokkos::Impl::g_device_cuda_lock_arrays.atomic[i] = 0;
  }
}

}  // namespace

namespace Impl {

CudaLockArrays g_host_cuda_lock_arrays = {nullptr, 0};

void initialize_host_cuda_lock_arrays() {
  desul::Impl::init_lock_arrays();
  desul::ensure_cuda_lock_arrays_on_device();

  if (g_host_cuda_lock_arrays.atomic != nullptr) return;
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMalloc(&g_host_cuda_lock_arrays.atomic,
                 sizeof(int) * (CUDA_SPACE_ATOMIC_MASK + 1)));
  Impl::cuda_device_synchronize(
      "Kokkos::Impl::initialize_host_cuda_lock_arrays: Pre Init Lock Arrays");
  g_host_cuda_lock_arrays.n = CudaInternal::concurrency();
  copy_cuda_lock_arrays_to_device();
  init_lock_array_kernel_atomic<<<(CUDA_SPACE_ATOMIC_MASK + 1 + 255) / 256,
                                  256>>>();
  Impl::cuda_device_synchronize(
      "Kokkos::Impl::initialize_host_cuda_lock_arrays: Post Init Lock Arrays");
}

void finalize_host_cuda_lock_arrays() {
  desul::Impl::finalize_lock_arrays();

  if (g_host_cuda_lock_arrays.atomic == nullptr) return;
  cudaFree(g_host_cuda_lock_arrays.atomic);
  g_host_cuda_lock_arrays.atomic = nullptr;
  g_host_cuda_lock_arrays.n      = 0;
#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
  copy_cuda_lock_arrays_to_device();
#endif
}

}  // namespace Impl

}  // namespace Kokkos

#else

void KOKKOS_CORE_SRC_CUDA_CUDA_LOCKS_PREVENT_LINK_ERROR() {}

#endif
