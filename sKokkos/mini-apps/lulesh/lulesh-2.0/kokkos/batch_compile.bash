make clean
make -j TEST_TARGET=1
make clean
make -j TEST_TARGET=2
make clean
make -j TEST_TARGET=3 TEST_SIZE=16
make clean
make -j TEST_TARGET=3 TEST_SIZE=32
make clean
make -j TEST_TARGET=3 TEST_SIZE=64
make clean
make -j TEST_TARGET=3 TEST_SIZE=128
make clean
make -j TEST_TARGET=3 TEST_SIZE=256
make clean
