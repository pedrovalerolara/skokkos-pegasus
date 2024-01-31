#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

#define GIGA_MEM 1024.0 * 1024.0 * 1024.0
#define GIGA_COMP 1000.0 * 1000.0 * 1000.0

#define DSIZE_LB 10
#define DSIZE_UB 1000
#define DSTEP 10
#define NITR 10

void set_arch( double operations );
void set_arch_reduce( double operations );

void set_arch( double operations )
{
  double time_cpu;
  double time_gpu;
  double latency_gpu;

  
  // ExCL equinox
  // Single precision
  // Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
  //time_cpu = operations / ( 1408.0 * (double) GIGA_COMP );
  // PCIe 3.0
  //latency_gpu = 4096.0 / ( 15.754 * (double) GIGA_MEM );
  // NVIDIA V100 GPU
  //time_gpu = ( operations / ( 14028.0 * (double) GIGA_COMP ) ) + latency_gpu;
  // ExCL Zenith
  // Double precision
  // AMD Ryzen Threadripper 3970X 32-Core https://en.wikichip.org/wiki/amd/microarchitectures/zen_2
  //time_cpu = operations / ( 2252.0 * (double) GIGA_COMP );
  // PCIe 4.0
  //latency_gpu = 4096.0 / ( 31.508 * (double) GIGA_MEM );
  // NVIDIA V100 GPU
  //time_gpu = ( operations / ( 35580.0 * (double) GIGA_COMP ) ) + latency_gpu;
  // Pegasus
  // Single precision
  // Intel(R) Xeon(R) Platinum 8468 (2.1GHz/48 Core)
  time_cpu = operations / ( 3225.6 * (double) GIGA_COMP );
  // PCIe 5.0
  latency_gpu = 4096.0 / ( 63.015 * (double) GIGA_MEM );
  // NVIDIA H100
  time_gpu = ( operations / ( 51200.0 * (double) GIGA_COMP ) ) + latency_gpu;

  //printf("Time CPU = %e, Time GPU = %e ->> Latency GPU = %e\n", time_cpu, time_gpu, latency_gpu);

  
  if ( time_cpu <= time_gpu )
  {
    acc_set_device_type(acc_device_host);
  }
  else
  {
    acc_set_device_type(acc_device_nvidia);
  }
}

void set_arch_reduce( double operations )
{
  double time_cpu;
  double time_gpu;
  double latency_gpu;
  //double latency_reduction;

  //Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz --> 20 cores
  //time_cpu = operations / ( 1408.0 * (double) GIGA_COMP ) + ( operations / 20.0 ) / ( 1408.0 * (double) GIGA_COMP );
  //latency_gpu = 4096.0 / ( 15.754 * (double) GIGA_MEM );
  //time_gpu = ( operations / ( 14028.0 * (double) GIGA_COMP ) ) + latency_gpu + ( operations / 84.0 ) / ( 14028.0 * (double) GIGA_COMP );
  //time_cpu = ( operations / 20.0 ) / ( 1408.0 * (double) GIGA_COMP );
  //latency_gpu = 4096.0 / ( 15.754 * (double) GIGA_MEM );
  //time_gpu = ( operations / 84.0 ) / ( 14028.0  * (double) GIGA_COMP );
  //time_cpu = operations / ( 1408.0 * (double) GIGA_COMP ) + ( operations / 20.0 ) / ( ( 1408.0 ) * (double) GIGA_COMP );
  //latency_gpu = 4096.0 / ( 15.754 * (double) GIGA_MEM );
  //time_gpu = ( operations / ( 14028.0 * (double) GIGA_COMP ) ) + latency_gpu + ( operations / 84.0 ) / ( ( 14028.0 ) * (double) GIGA_COMP );

  // ExCL equinox
  // Single precision
  //Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz --> 20 cores
  //time_cpu = operations / ( 1408.0 * (double) GIGA_COMP ) + ( operations / 20.0 ) / ( ( 1408.0 / 20.0 ) * (double) GIGA_COMP );
  // PCIe 3.0
  //latency_gpu = 4096.0 / ( 15.754 * (double) GIGA_MEM );
  // NVIDIA V100 GPU
  //time_gpu = ( operations / ( 14028.0 * (double) GIGA_COMP ) ) + latency_gpu + ( operations / 84.0 ) / ( ( 14028.0 / 84.0 ) * (double) GIGA_COMP );
  //printf("Time CPU = %e, Time GPU = %e ->> Latency GPU = %e\n", time_cpu, time_gpu, latency_gpu);

  // ExCL Zenith
  // Single precision
  // AMD Ryzen Threadripper 3970X 32-Core https://en.wikichip.org/wiki/amd/microarchitectures/zen_2
  //time_cpu = operations / ( 2252.0 * (double) GIGA_COMP ) + ( operations / 32.0 ) / ( ( 2252.0 / 32.0 ) * (double) GIGA_COMP );
  // PCIe 4.0
  //latency_gpu = 4096.0 / ( 31.508 * (double) GIGA_MEM );
  //latency_reduction = ( 168.0 * sizeof(float) ) / ( 31.508 * (double) GIGA_MEM );
  // NVIDIA GeForce RTX 3090 GPU
  //time_gpu = ( operations / ( 35580.0 * (double) GIGA_COMP ) ) + latency_gpu + ( operations / 180.0 ) / ( ( 35580.0 / 180.0 ) * (double) GIGA_COMP );// + latency_reduction;

  // Pegasus
  // Single precision
  // Intel(R) Xeon(R) Platinum 8468 (2.1GHz/48 Core)
  time_cpu = operations / ( 3225.6 * (double) GIGA_COMP ) + ( operations / 48.0 ) / ( ( 2252.0 / 48.0 ) * (double) GIGA_COMP );;
  // PCIe 5.0
  latency_gpu = 4096.0 / ( 63.015 * (double) GIGA_MEM );
  // NVIDIA H100
  time_gpu = ( operations / ( 51200.0 * (double) GIGA_COMP ) ) + latency_gpu + ( operations / 144.0 ) / ( ( 51200.0 / 144.0 ) * (double) GIGA_COMP );

  if ( time_cpu <= time_gpu )
  {
    acc_set_device_type(acc_device_host);
  }
  else
  {
    acc_set_device_type(acc_device_nvidia);
  }
}



int main( int argc, char* argv[] )
{

  struct timeval start, end;
  double time; 
	  
  int M;
  int N;

  Kokkos::initialize( argc, argv );
  {
  
  for (int i = DSIZE_LB; i <= DSIZE_UB; i += DSTEP )
  {
  
  M = i;
  N = i;
  //acc_set_device_type(acc_device_nvidia);
  acc_set_device_type(acc_device_host);
  //set_arch(2.0*M*N);
  //set_arch_reduce(2.0*M*N);
 
  //auto X  = static_cast<float*>(Kokkos::kokkos_malloc<>(M * N * sizeof(float)));
  //auto Y  = static_cast<float*>(Kokkos::kokkos_malloc<>(M * N * sizeof(float)));
  Kokkos::View<double**, Kokkos::LayoutRight> X("X", M , N);
  Kokkos::View<double**, Kokkos::LayoutRight> Y("Y", M , N);

  typedef Kokkos::MDRangePolicy< Kokkos::Rank<2> > mdrange_policy;

  //printf("AXPY -- kokkos parallel_for\n");
  Kokkos::parallel_for( "axpy_init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    //X[m * N + n] = 2.0;
    //Y[m * N + n] = 2.0;
    X(m, n) = 2.0;
    Y(m, n) = 2.0;
    //printf("X[%d] = %2.f and Y[%d] = %2.f\n", m, X[m], m, Y[m]);
  });

  Kokkos::fence();

  // Warming
  Kokkos::parallel_for( "axpy_comp", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    float alpha = 2.0;
    //Y[m * N + n] += alpha * X[m * N + n];
    Y(m, n) += alpha * X(m, n);
    //printf("Y[%d] = %2.f\n", m, Y[m]);
  });
  
  Kokkos::fence();

  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < NITR; i++ )
  { 

  Kokkos::parallel_for( "axpy_comp", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    float alpha = 2.0;
    //Y[m * N + n] += alpha * X[m * N + n];
    Y(m, n) += alpha * X(m, n);
    //printf("Y[%d] = %2.f\n", m, Y[m]);
  });
 
  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / NITR;

  //printf( "AXPY = %d Time ( %e s )\n", M, time );
  //printf( "%d\n", M );
  printf( "%e\n", time );

  /*
  //printf("DOT Product -- Kokkos parallel_reduce\n");
  Kokkos::parallel_for( "dotproduct_init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    X[m * N + n] = 2.0;
    Y[m * N + n] = 2.0;
    //printf("X[%d] = %2.f and Y[%d] = %2.f\n", m, X[m], m, Y[m]);
  });

  float result;

  //Warming
  Kokkos::parallel_reduce( "dotproduct_comp", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, float &update )
  {
    update += X[m * N + n] * Y[m * N + n];
  }, result );
 
  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < NITR; i++ )
  { 
  Kokkos::parallel_reduce( "dotproduct_comp", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, float &update )
  {
    update += X[m * N + n] * Y[m * N + n];
  }, result );
 
  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / 100.0;

  //printf( "DOT = %d Time ( %e s )\n", M, time );
  printf( "%e\n", time );
  //printf( "%d\n", M );

  //printf("DOT Product result %2.f\n", result);

  //Kokkos::kokkos_free<>(X);
  //Kokkos::kokkos_free<>(Y);
  */

  }
  
  }

  Kokkos::finalize();
  
  return 0;
}
