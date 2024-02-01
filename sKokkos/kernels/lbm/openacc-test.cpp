#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <cmath>

#define GIGA_MEM 1024.0 * 1024.0 * 1024.0
#define GIGA_COMP 1000.0 * 1000.0 * 1000.0
#define SIZE 160

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

  M = SIZE;
  N = SIZE;
 
  set_arch(M*N*335.0);
  acc_set_device_type(acc_device_nvidia);
  //acc_set_device_type(acc_device_host);

  auto f1  = static_cast<float*>(Kokkos::kokkos_malloc<>(M * N * 9 * sizeof(float)));
  auto f2  = static_cast<float*>(Kokkos::kokkos_malloc<>(M * N * 9 * sizeof(float)));
  auto cx  = static_cast<int*>(Kokkos::kokkos_malloc<>(9 * sizeof(int)));
  auto cy  = static_cast<int*>(Kokkos::kokkos_malloc<>(9 * sizeof(int)));
  auto w  = static_cast<float*>(Kokkos::kokkos_malloc<>(9 * sizeof(float)));

  Kokkos::parallel_for( "lbm_init", 9 , KOKKOS_LAMBDA ( int m )
  {
    if ( m == 0 ){
      cx[0] = 0;
      cy[0] = 0;
    }
    else if ( m == 1 ){
      cx[1] = 1;
      cy[1] = 0;
    }
    else if ( m == 2 ){
      cx[2] = -1;
      cy[2] = 0;
    }
    else if ( m == 3 ){
      cx[3] = 0;
      cy[3] = 1;
    }
    else if ( m == 4 ){
      cx[4] = 0;
      cy[4] = -1;
    }
    else if ( m == 5 ){
      cx[5] = 1;
      cy[5] = 1;
    }
    else if ( m == 6 ){
      cx[6] = -1;
      cy[6] = 1;
    }
    else if ( m == 7 ){
      cx[7] = -1;
      cy[7] = -1;
    }
    else if ( m == 8 ){
      cx[8] = 1;
      cy[8] = -1;
    }
    w[m] = 1.0;
  });

  Kokkos::parallel_for( "lbm", M*N, KOKKOS_LAMBDA ( int m )
  {
    int x = floor( m / SIZE ); 
    int y = m % SIZE;
    int x_stream, y_stream, ind;
    float u, v, p = 0.0;
    float feq, cu;
    float f[9];
    float t = 1.0;
    int k = 0;

    if ( x > 1 && x < SIZE - 2 && y > 1 && y < SIZE - 2 )  
    {
        for ( k = 0; k < 9; k++ )
	{
          x_stream = x - cx[k];
          y_stream = y - cy[k];
          ind = ( k * SIZE * SIZE ) + ( x_stream * SIZE ) + y_stream;
          f[k] = f1[ind];
	} 
        for ( k = 0; k < 9; k++ )
	{
          p += f[k];
          u += f[k] * cx[k];
          v += f[k] * cy[k];
	}
        u = u / p;
        v = v / p;
        for ( k = 0; k < 9; k++ )
        {
          cu = cx[k] * u + cy[k] * v;
          feq = w[k] * p * ( 1.0 + 3.0 * cu + cu * cu - 1.5 * ( ( u * u ) + ( v * v ) ) );
          ind = ( k * SIZE * SIZE ) + ( x * SIZE ) + y;
          f2[ind] = f[k] * (1.0 - 1.0 / t) + feq * 1.0 / t;
	}
      }
  } );
 
  Kokkos::fence();


  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < 10; i++ )
  { 

  Kokkos::parallel_for( "lbm", M*N, KOKKOS_LAMBDA ( int m )
  {
    int x = floor( m / SIZE ); 
    int y = m % SIZE;
    int x_stream, y_stream, ind;
    float u, v, p = 0.0;
    float feq, cu;
    float f[9];
    float t = 1.0;
    int k = 0;

    if ( x > 1 && x < SIZE - 2 && y > 1 && y < SIZE - 2 )  
    {
        for ( k = 0; k < 9; k++ )
	{
          x_stream = x - cx[k];
          y_stream = y - cy[k];
          ind = ( k * SIZE * SIZE ) + ( x_stream * SIZE ) + y_stream;
          f[k] = f1[ind];
	} 
        for ( k = 0; k < 9; k++ )
	{
          p += f[k];
          u += f[k] * cx[k];
          v += f[k] * cy[k];
	}
        u = u / p;
        v = v / p;
        for ( k = 0; k < 9; k++ )
        {
          cu = cx[k] * u + cy[k] * v;
          feq = w[k] * p * ( 1.0 + 3.0 * cu + cu * cu - 1.5 * ( ( u * u ) + ( v * v ) ) );
          ind = ( k * SIZE * SIZE ) + ( x * SIZE ) + y;
          f2[ind] = f[k] * (1.0 - 1.0 / t) + feq * 1.0 / t;
	}
      }
  } );
 
  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / 10.0;

  //printf( "LBM = %d Time ( %e s )\n", M, time );
  printf( "%e\n", time );

  Kokkos::kokkos_free<>(f1);
  Kokkos::kokkos_free<>(f2);

  }
  
  Kokkos::finalize();

  return 0;
}
