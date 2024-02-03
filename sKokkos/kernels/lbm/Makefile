KOKKOS_PATH = ../../skokkos/
KOKKOS_DEVICES = "OpenACC"
EXE_NAME = "OpenACC"
TEST_TARGET ?= 1
TEST_SIZE ?= 160
ifeq ($(TEST_TARGET),1)
TARGET_DEVICE=cpu
else ifeq ($(TEST_TARGET),2)
TARGET_DEVICE=gpu
else
TARGET_DEVICE=auto
endif

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

CXX = nvc++ 
#CXX = g++ 
CXXFLAGS = -O3 -acc=gpu,multicore -DTEST_TARGET=${TEST_TARGET} -DSIZE=${TEST_SIZE}
#CXXFLAGS = -O3 -DTEST_TARGET=${TEST_TARGET} -DSIZE=${TEST_SIZE}
#CXXFLAGS = -O3 -acc=gpu,multicore #-tp=pwr9
#CXXFLAGS = -O3 -Minfo=acc -acc
#CXXFLAGS = -O3 -Minfo=acc -acc -gpu=keep
LINK = ${CXX}
LINKFLAGS = -O3 -acc=gpu,multicore -DTEST_TARGET=${TEST_TARGET} -DSIZE=${TEST_SIZE}
#LINKFLAGS = -O3 -DTEST_TARGET=${TEST_TARGET} -DSIZE=${TEST_SIZE}
#LINKFLAGS = -acc=gpu,multicore #-tp=pwr9
EXE = ${EXE_NAME}.acc.${TARGET_DEVICE}.s${TEST_SIZE}
#EXE = ${EXE_NAME}.mp
#KOKKOS_ARCH = "Volta70"
#KOKKOS_ARCH = "BDW"
#KOKKOS_ARCH = "Power9"
#KOKKOS_ARCH = "Zen2"
KOKKOS_ARCH = "BDW,Hopper90"
KOKKOS_CUDA_OPTIONS = "enable_lambda"

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -rf *.o desul

purge: clean
	rm -f OpenACC.acc.cpu* OpenACC.acc.gpu* OpenACC.acc.auto*

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)

