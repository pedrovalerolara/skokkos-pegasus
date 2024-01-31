#!/bin/bash
module load nvhpc/22.11
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export TARGET=OpenACC
export KOKKOS_PATH=../../../../hkokkacc/
