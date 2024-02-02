#Set your Kokkos path to something appropriate
KOKKOS_PATH = ../../../../skokkos/
KOKKOS_DEVICES ?= "OpenACC"
#KOKKOS_ARCH = "Volta70"
#KOKKOS_ARCH = "BDW"
#KOKKOS_ARCH = "Zen2"
KOKKOS_ARCH = "BDW,Hopper90"
KOKKOS_CUDA_OPTIONS = enable_lambda
KOKKOS_OPTIONS = disable_desul_atomics
TEST_TARGET ?= 1
TEST_SIZE ?= 16

SRC = $(wildcard *.cc)

default: build
	echo "Start Build"
	
#CXXFLAGS = -g -O3 -acc=gpu,multicore -DUSE_OMP=1 -DRAJA_USE_GETTIME -I./includes -I./ 
CXXFLAGS = -O3 -DUSE_OMP=1 -DRAJA_USE_GETTIME -I./includes -I./ -DTEST_TARGET=${TEST_TARGET} -DTEST_SIZE=${TEST_SIZE}

ifeq ( $(USE_MPI), yes) 
CXX = mpicxx
CXXFLAGS += -DUSE_MPI=1
else
CXX ?= nvc++
CXXFLAGS += -DUSE_MPI=0
KOKKOS_CXX_STANDARD = c++17
endif

LINK = ${CXX}

LINKFLAGS =  
ifeq ($(TEST_TARGET),1)
TARGET_DEVICE=cpu
else ifeq ($(TEST_TARGET),2)
TARGET_DEVICE=gpu
else
TARGET_DEVICE=auto
endif
ifeq ($(TEST_TARGET),3)
EXE = lulesh.kk.${TARGET_DEVICE}.s${TEST_SIZE}
else
EXE = lulesh.kk.${TARGET_DEVICE}
endif

DEPFLAGS = -M

OBJ = $(SRC:.cc=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o 

purge: clean
	rm -f lulesh.kk.*

# Compilation rules

%.o:%.cc $(KOKKOS_CPP_DEPENDS) lulesh.h lulesh_tuple.h
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<



