#!/bin/bash
module load nvhpc/22.11
export KOKKOS_ROOT=../../../../hkokkacc/
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
