make clean
make -j TEST_TARGET=1 TEST_APP=1
make clean
make -j TEST_TARGET=1 TEST_APP=2
make clean
make -j TEST_TARGET=2 TEST_APP=1
make clean
make -j TEST_TARGET=2 TEST_APP=2
make clean
make -j TEST_TARGET=3 TEST_APP=1
make clean
make -j TEST_TARGET=3 TEST_APP=2
make clean
