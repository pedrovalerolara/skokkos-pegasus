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

#ifndef KOKKOS_BIT_MANIPULATION_HPP
#define KOKKOS_BIT_MANIPULATION_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace Kokkos::Impl {

template <class T>
KOKKOS_FUNCTION constexpr int countl_zero_fallback(T x) {
  // From Hacker's Delight (2nd edition) section 5-3
  unsigned int y = 0;
  using ::Kokkos::Experimental::digits_v;
  int n = digits_v<T>;
  int c = digits_v<T> / 2;
  do {
    y = x >> c;
    if (y != 0) {
      n -= c;
      x = y;
    }
    c >>= 1;
  } while (c != 0);
  return n - static_cast<int>(x);
}

template <class T>
KOKKOS_FUNCTION constexpr int countr_zero_fallback(T x) {
  using ::Kokkos::Experimental::digits_v;
  return digits_v<T> - countl_zero_fallback(static_cast<T>(
                           static_cast<T>(~x) & static_cast<T>(x - 1)));
}

template <class T>
KOKKOS_FUNCTION constexpr int popcount_fallback(T x) {
  int c = 0;
  for (; x != 0; x &= x - 1) {
    ++c;
  }
  return c;
}

template <class T>
inline constexpr bool is_standard_unsigned_integer_type_v =
    std::is_same_v<T, unsigned char> || std::is_same_v<T, unsigned short> ||
    std::is_same_v<T, unsigned int> || std::is_same_v<T, unsigned long> ||
    std::is_same_v<T, unsigned long long>;

}  // namespace Kokkos::Impl

namespace Kokkos {

//<editor-fold desc="[bit.count], counting">
template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, int>
countl_zero(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  if (x == 0) return digits_v<T>;
  // TODO use compiler intrinsics when available
  return Impl::countl_zero_fallback(x);
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, int>
countl_one(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  using ::Kokkos::Experimental::finite_max_v;
  if (x == finite_max_v<T>) return digits_v<T>;
  return countl_zero(static_cast<T>(~x));
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, int>
countr_zero(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  if (x == 0) return digits_v<T>;
  // TODO use compiler intrinsics when available
  return Impl::countr_zero_fallback(x);
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, int>
countr_one(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  using ::Kokkos::Experimental::finite_max_v;
  if (x == finite_max_v<T>) return digits_v<T>;
  return countr_zero(static_cast<T>(~x));
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, int>
popcount(T x) noexcept {
  if (x == 0) return 0;
  // TODO use compiler intrinsics when available
  return Impl::popcount_fallback(x);
}
//</editor-fold>

//<editor-fold desc="[bit.pow.two], integral powers of 2">
template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, bool>
has_single_bit(T x) noexcept {
  return x != 0 && (((x & (x - 1)) == 0));
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, T>
bit_ceil(T x) noexcept {
  if (x <= 1) return 1;
  using ::Kokkos::Experimental::digits_v;
  return T{1} << (digits_v<T> - countl_zero(static_cast<T>(x - 1)));
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, T>
bit_floor(T x) noexcept {
  if (x == 0) return 0;
  using ::Kokkos::Experimental::digits_v;
  return T{1} << (digits_v<T> - 1 - countl_zero(x));
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, T>
bit_width(T x) noexcept {
  if (x == 0) return 0;
  using ::Kokkos::Experimental::digits_v;
  return digits_v<T> - countl_zero(x);
}
//</editor-fold>

}  // namespace Kokkos

namespace Kokkos::Impl {

#if defined(KOKKOS_COMPILER_CLANG) || defined(KOKKOS_COMPILER_GCC)
#define KOKKOS_IMPL_USE_GCC_BUILT_IN_FUNCTIONS
#endif

template <class T>
KOKKOS_IMPL_DEVICE_FUNCTION
    std::enable_if_t<is_standard_unsigned_integer_type_v<T>, int>
    countl_zero_builtin_device(T x) noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if constexpr (sizeof(T) == sizeof(long long int)) {
    return __clzll(reinterpret_cast<long long int&>(x));
  } else if constexpr (sizeof(T) == sizeof(int)) {
    return __clz(reinterpret_cast<int&>(x));
  } else {
    using ::Kokkos::Experimental::digits_v;
    constexpr int shift = digits_v<unsigned int> - digits_v<T>;
    return __clz(x) - shift;
  }
#elif defined(KOKKOS_ENABLE_SYCL)
  return sycl::clz(x);
#else
  return countl_zero_fallback(x);
#endif
}

template <class T>
KOKKOS_IMPL_HOST_FUNCTION
    std::enable_if_t<is_standard_unsigned_integer_type_v<T>, int>
    countl_zero_builtin_host(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  if (x == 0) return digits_v<T>;
#ifdef KOKKOS_IMPL_USE_GCC_BUILT_IN_FUNCTIONS
  if constexpr (std::is_same_v<T, unsigned long long>) {
    return __builtin_clzll(x);
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __builtin_clzl(x);
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __builtin_clz(x);
  } else {
    constexpr int shift = digits_v<unsigned int> - digits_v<T>;
    return __builtin_clz(x) - shift;
  }
#else
  return countl_zero_fallback(x);
#endif
}

template <class T>
KOKKOS_IMPL_DEVICE_FUNCTION
    std::enable_if_t<is_standard_unsigned_integer_type_v<T>, int>
    countr_zero_builtin_device(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  if (x == 0) return digits_v<T>;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if constexpr (sizeof(T) == sizeof(long long int)) {
    return __ffsll(reinterpret_cast<long long int&>(x)) - 1;
  } else {
    return __ffs(reinterpret_cast<int&>(x)) - 1;
  }
#elif defined(KOKKOS_ENABLE_SYCL)
  return sycl::ctz(x);
#else
  return countr_zero_fallback(x);
#endif
}

template <class T>
KOKKOS_IMPL_HOST_FUNCTION
    std::enable_if_t<is_standard_unsigned_integer_type_v<T>, int>
    countr_zero_builtin_host(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  if (x == 0) return digits_v<T>;
#ifdef KOKKOS_IMPL_USE_GCC_BUILT_IN_FUNCTIONS
  if constexpr (std::is_same_v<T, unsigned long long>) {
    return __builtin_ctzll(x);
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __builtin_ctzl(x);
  } else {
    return __builtin_ctz(x);
  }
#else
  return countr_zero_fallback(x);
#endif
}

template <class T>
KOKKOS_IMPL_DEVICE_FUNCTION
    std::enable_if_t<is_standard_unsigned_integer_type_v<T>, int>
    popcount_builtin_device(T x) noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if constexpr (sizeof(T) == sizeof(long long int)) {
    return __popcll(x);
  } else {
    return __popc(x);
  }
#elif defined(KOKKOS_ENABLE_SYCL)
  return sycl::popcount(x);
#else
  return popcount_fallback(x);
#endif
}

template <class T>
KOKKOS_IMPL_HOST_FUNCTION
    std::enable_if_t<is_standard_unsigned_integer_type_v<T>, int>
    popcount_builtin_host(T x) noexcept {
#ifdef KOKKOS_IMPL_USE_GCC_BUILT_IN_FUNCTIONS
  if constexpr (std::is_same_v<T, unsigned long long>) {
    return __builtin_popcountll(x);
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __builtin_popcountl(x);
  } else {
    return __builtin_popcount(x);
  }
#else
  return popcount_fallback(x);
#endif
}

#undef KOKKOS_IMPL_USE_GCC_BUILT_IN_FUNCTIONS

}  // namespace Kokkos::Impl

namespace Kokkos::Experimental {

template <class T>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::Impl::is_standard_unsigned_integer_type_v<T>, int>
countl_zero_builtin(T x) noexcept {
  KOKKOS_IF_ON_DEVICE((return ::Kokkos::Impl::countl_zero_builtin_device(x);))
  KOKKOS_IF_ON_HOST((return ::Kokkos::Impl::countl_zero_builtin_host(x);))
}

template <class T>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::Impl::is_standard_unsigned_integer_type_v<T>, int>
countl_one_builtin(T x) noexcept {
  if (x == finite_max_v<T>) return digits_v<T>;
  return countl_zero_builtin(static_cast<T>(~x));
}

template <class T>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::Impl::is_standard_unsigned_integer_type_v<T>, int>
countr_zero_builtin(T x) noexcept {
  KOKKOS_IF_ON_DEVICE((return ::Kokkos::Impl::countr_zero_builtin_device(x);))
  KOKKOS_IF_ON_HOST((return ::Kokkos::Impl::countr_zero_builtin_host(x);))
}

template <class T>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::Impl::is_standard_unsigned_integer_type_v<T>, int>
countr_one_builtin(T x) noexcept {
  if (x == finite_max_v<T>) return digits_v<T>;
  return countr_zero_builtin(static_cast<T>(~x));
}

template <class T>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::Impl::is_standard_unsigned_integer_type_v<T>, int>
popcount_builtin(T x) noexcept {
  KOKKOS_IF_ON_DEVICE((return ::Kokkos::Impl::popcount_builtin_device(x);))
  KOKKOS_IF_ON_HOST((return ::Kokkos::Impl::popcount_builtin_host(x);))
}

template <class T>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::Impl::is_standard_unsigned_integer_type_v<T>, bool>
has_single_bit_builtin(T x) noexcept {
  return has_single_bit(x);  // no benefit to call the _builtin variant
}

template <class T>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::Impl::is_standard_unsigned_integer_type_v<T>, T>
    bit_ceil_builtin(T x) noexcept {
  if (x <= 1) return 1;
  return T{1} << (digits_v<T> - countl_zero_builtin(static_cast<T>(x - 1)));
}

template <class T>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::Impl::is_standard_unsigned_integer_type_v<T>, T>
    bit_floor_builtin(T x) noexcept {
  if (x == 0) return 0;
  return T{1} << (digits_v<T> - 1 - countl_zero_builtin(x));
}

template <class T>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::Impl::is_standard_unsigned_integer_type_v<T>, T>
    bit_width_builtin(T x) noexcept {
  if (x == 0) return 0;
  return digits_v<T> - countl_zero_builtin(x);
}

}  // namespace Kokkos::Experimental

#endif
