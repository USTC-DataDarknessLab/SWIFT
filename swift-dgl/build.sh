#!/bin/bash
# conda activate dc_test

rm -rf build 
mkdir build 
cd build
cmake -DBUILD_TORCH=ON -DUSE_CUDA=ON -DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7 ..
make -j4
cd ../python && python setup.py install > /dev/null
