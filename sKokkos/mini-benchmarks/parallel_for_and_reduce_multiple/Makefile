KOKKOS_PATH = ../../skokkos/
#KOKKOS_DEVICES = "OpenACC"
EXE_NAME = "OpenACC"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

CXX = nvc++ 
#CXX = g++ 
#CXXFLAGS = -O3 #-acc=gpu,multicore #-tp=pwr9
CXXFLAGS = -O3 -acc=gpu,multicore #-tp=pwr9
#CXXFLAGS = -O3 -Minfo=acc -acc
#CXXFLAGS = -O3 -Minfo=acc -acc -gpu=keep
LINK = ${CXX}
LINKFLAGS = -acc=gpu,multicore #-tp=pwr9
#LINKFLAGS = -acc=gpu,multicore #-tp=pwr9
EXE = ${EXE_NAME}.acc
#EXE = ${EXE_NAME}.mp
#KOKKOS_ARCH = "Volta70"
#KOKKOS_ARCH = "Power9"
KOKKOS_ARCH = "BDW"
#KOKKOS_ARCH = "Zen2"
KOKKOS_CUDA_OPTIONS = "enable_lambda"

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.acc *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)

