
//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>

#include <miniFE_version.h>

#include <outstream.hpp>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <Box.hpp>
#include <BoxPartition.hpp>
#include <box_utils.hpp>
#include <Parameters.hpp>
#include <utils.hpp>
#include <driver.hpp>
#include <YAML_Doc.hpp>
#include <Kokkos_Types.hpp>
#if MINIFE_INFO != 0
#include <miniFE_info.hpp>
#else
#include <miniFE_no_info.hpp>
#endif

//TEST_TARGET = 1 for CPU test
//              2 for GPU test
//              3 for autotuning
#if !defined(TEST_TARGET)
#define TEST_TARGET 1
#endif
#if !defined(TEST_SIZE)
#define TEST_SIZE 16
#endif

//The following macros should be specified as compile-macros in the
//makefile. They are defaulted here just in case...
#ifndef MINIFE_SCALAR
//#define MINIFE_SCALAR double
#define MINIFE_SCALAR float
#endif
#ifndef MINIFE_LOCAL_ORDINAL
#define MINIFE_LOCAL_ORDINAL int
#endif
#ifndef MINIFE_GLOBAL_ORDINAL
#define MINIFE_GLOBAL_ORDINAL int
#endif

// ************************************************************************

void add_params_to_yaml(YAML_Doc& doc, miniFE::Parameters& params);
void add_configuration_to_yaml(YAML_Doc& doc, int numprocs);
void add_timestring_to_yaml(YAML_Doc& doc);

//
//We will create a 'box' of size nx X ny X nz, partition it among processors,
//then call miniFE::driver which will use the partitioned box as the domain
//from which to assemble finite-element matrices into a global matrix and
//vector, then solve the linear-system using Conjugate Gradients.
//


#define GIGA_MEM 1024.0 * 1024.0 * 1024.0
#define GIGA_COMP 1000.0 * 1000.0 * 1000.0

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
  //latency_gpu = 4096.0 / ( 63.015 * (double) GIGA_MEM );
  latency_gpu = 32768.0 / ( 63.015 * (double) GIGA_MEM );
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
  //latency_gpu = 4096.0 / ( 63.015 * (double) GIGA_MEM );
  latency_gpu = 16384.0 / ( 63.015 * (double) GIGA_MEM );
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

int main(int argc, char** argv) {
#if TEST_TARGET == 1
  //printf("==> target device: CPU\n");
#elif TEST_TARGET == 2
  //printf("==> target device: GPU\n");
#else
  //printf("==> target device: AUTO\n");
#endif
  
  miniFE::Parameters params;
  miniFE::get_parameters(argc, argv, params);

  printf("Nz = %d, Ny = %d, Nz = %d\n", params.nx, params.ny, params.nz);

  int numprocs = 1, myproc = 0;
  miniFE::initialize_mpi(argc, argv, numprocs, myproc);

  Kokkos::initialize(argc,argv);
  
  // We use number of non zeros elements (NNZ) for the tuning factor
  // depending on the size of the 3D domain, there is a different NNZ
  // I already computed those NNZ, so you can use the lines below
#if TEST_TARGET == 1
  acc_set_device_type(acc_device_host);
#elif TEST_TARGET == 2
  acc_set_device_type(acc_device_nvidia);
#else
#if TEST_SIZE == 16
  set_arch( (double) 117649.0 ); //NNZ for 16x16x16
#elif TEST_SIZE == 32
  set_arch( (double) 912673.0 ); //NNZ for 32x32x32
#elif TEST_SIZE == 64
  set_arch( (double) 7189057.0 ); //NNZ for 64x64x64
#elif TEST_SIZE == 128
  set_arch( (double) 57066625.0 ); //NNZ for 128x128x182
#else
  printf("==> [ERROR] unknown TEST_SIZE value: %d\n", TEST_SIZE);
  return 0;
#endif
#endif
  
  // MGT
  void initTimeProbes();
  initTimeProbes();


  if(myproc==0) {
    std::cout << "MiniFE Mini-App, Kokkos Peer Implementation" << std::endl;
  }

  miniFE::timer_type start_time = miniFE::mytimer();

#ifdef MINIFE_DEBUG
  outstream(numprocs, myproc);
#endif

  //make sure each processor has the same parameters:
  miniFE::broadcast_parameters(params);

  Box global_box = { 0, params.nx, 0, params.ny, 0, params.nz };
  std::vector<Box> local_boxes(numprocs);

  box_partition(0, numprocs, 2, global_box, &local_boxes[0]);

  Box& my_box = local_boxes[myproc];

  MINIFE_GLOBAL_ORDINAL num_my_ids = miniFE::get_num_ids<MINIFE_GLOBAL_ORDINAL>(my_box);
  MINIFE_GLOBAL_ORDINAL min_ids = num_my_ids;

#ifdef HAVE_MPI
  MPI_Datatype mpi_dtype = miniFE::TypeTraits<MINIFE_GLOBAL_ORDINAL>::mpi_type();
  MPI_Allreduce(&num_my_ids, &min_ids, 1, mpi_dtype, MPI_MIN, MPI_COMM_WORLD);
#endif

  if (min_ids == 0) {
    std::cout<<"One or more processors have 0 equations. Not currently supported. Exiting."<<std::endl;

    miniFE::finalize_mpi();

    return 1;
  }

  std::ostringstream osstr;
  osstr << "miniFE." << params.nx << "x" << params.ny << "x" << params.nz;
#ifdef HAVE_MPI
  osstr << ".P"<<numprocs;
#endif
  osstr << ".";
  if (params.name != "") osstr << params.name << ".";

  YAML_Doc doc("miniFE", MINIFE_VERSION, ".", osstr.str());
  if (myproc == 0) {
    add_params_to_yaml(doc, params);
    add_configuration_to_yaml(doc, numprocs);
    add_timestring_to_yaml(doc);
  }

  //Most of the program is performed in the 'driver' function, which is
  //templated on < Scalar, LocalOrdinal, GlobalOrdinal >.
  //To run miniFE with float instead of double, or 'long long' instead of int,
  //etc., change these template-parameters by changing the macro definitions in
  //the makefile or on the make command-line.

  int return_code =
     miniFE::driver< MINIFE_SCALAR, MINIFE_LOCAL_ORDINAL, MINIFE_GLOBAL_ORDINAL>(global_box, my_box, params, doc);

  miniFE::timer_type total_time = miniFE::mytimer() - start_time;

  if (myproc == 0) {
    doc.add("Total Program Time",total_time);
    doc.generateYAML();
  }
  // MGT
  void dumpTimers();
  dumpTimers();

  Kokkos::finalize();

  miniFE::finalize_mpi();

  return return_code;
}

void add_params_to_yaml(YAML_Doc& doc, miniFE::Parameters& params)
{
  doc.add("Global Run Parameters","");
  doc.get("Global Run Parameters")->add("dimensions","");
  doc.get("Global Run Parameters")->get("dimensions")->add("nx",params.nx);
  doc.get("Global Run Parameters")->get("dimensions")->add("ny",params.ny);
  doc.get("Global Run Parameters")->get("dimensions")->add("nz",params.nz);
  doc.get("Global Run Parameters")->add("load_imbalance", params.load_imbalance);
  if (params.mv_overlap_comm_comp == 1) {
    std::string val("1 (yes)");
    doc.get("Global Run Parameters")->add("mv_overlap_comm_comp", val);
  }
  else {
    std::string val("0 (no)");
    doc.get("Global Run Parameters")->add("mv_overlap_comm_comp", val);
  }
}

void add_configuration_to_yaml(YAML_Doc& doc, int numprocs)
{
  doc.get("Global Run Parameters")->add("number of processors", numprocs);

  doc.add("Platform","");
  doc.get("Platform")->add("hostname",MINIFE_HOSTNAME);
  doc.get("Platform")->add("kernel name",MINIFE_KERNEL_NAME);
  doc.get("Platform")->add("kernel release",MINIFE_KERNEL_RELEASE);
  doc.get("Platform")->add("processor",MINIFE_PROCESSOR);

  doc.add("Build","");
  doc.get("Build")->add("CXX",MINIFE_CXX);
#if MINIFE_INFO != 0
  doc.get("Build")->add("compiler version",MINIFE_CXX_VERSION);
#endif
  doc.get("Build")->add("CXXFLAGS",MINIFE_CXXFLAGS);
  std::string using_mpi("no");
#ifdef HAVE_MPI
  using_mpi = "yes";
#endif
  doc.get("Build")->add("using MPI",using_mpi);
}

void add_timestring_to_yaml(YAML_Doc& doc)
{
  std::time_t rawtime;
  struct tm * timeinfo;
  std::time(&rawtime);
  timeinfo = std::localtime(&rawtime);
  std::ostringstream osstr;
  osstr.fill('0');
  osstr << timeinfo->tm_year+1900 << "-";
  osstr.width(2); osstr << timeinfo->tm_mon+1 << "-";
  osstr.width(2); osstr << timeinfo->tm_mday << ", ";
  osstr.width(2); osstr << timeinfo->tm_hour << "-";
  osstr.width(2); osstr << timeinfo->tm_min << "-";
  osstr.width(2); osstr << timeinfo->tm_sec;
  std::string timestring = osstr.str();
  doc.add("Run Date/Time",timestring);
}

