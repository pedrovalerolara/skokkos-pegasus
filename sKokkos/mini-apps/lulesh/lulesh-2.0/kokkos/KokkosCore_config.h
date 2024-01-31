/* ---------------------------------------------
Makefile constructed configuration:
----------------------------------------------*/
#if !defined(KOKKOS_MACROS_HPP) || defined(KOKKOS_CORE_CONFIG_H)
#error "Do not include KokkosCore_config.h directly; include Kokkos_Macros.hpp instead."
#else
#define KOKKOS_CORE_CONFIG_H
#endif

#define KOKKOS_VERSION 40099
#define KOKKOS_VERSION_MAJOR 4
#define KOKKOS_VERSION_MINOR 0
#define KOKKOS_VERSION_PATCH 99

/* Execution Spaces */
#define KOKKOS_ENABLE_OPENACC
#define KOKKOS_ENABLE_SERIAL
/* General Settings */
#define KOKKOS_ENABLE_DEPRECATED_CODE_4
#define KOKKOS_ENABLE_CXX17
#define KOKKOS_ENABLE_COMPLEX_ALIGN
#define KOKKOS_ENABLE_LIBDL
/* Optimization Settings */
/* Cuda Settings */
#define KOKKOS_ARCH_AMD_ZEN2
#define KOKKOS_ARCH_AMD_AVX2
#define KOKKOS_ENABLE_IMPL_MDSPAN
