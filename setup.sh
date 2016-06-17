#!/bin/bash

BUILD_DIR=build/RAJA
ABS_PATH=$(dirname $(readlink -f $0))
RAJA_INSTALL_DIR=$ABS_PATH/dist/RAJA

rm -rf $BUILD_DIR 2>/dev/null
mkdir -p $BUILD_DIR && pushd $BUILD_DIR

RAJA_DIR=$(git rev-parse --show-toplevel)/RAJA

cmake -C ${RAJA_DIR}/host-configs/chaos/clang_3_7_0.cmake \
      -DCMAKE_INSTALL_PREFIX=${RAJA_INSTALL_DIR} -DRAJA_ENABLE_TESTS=0\
      "$@" ${RAJA_DIR}

popd

make -C $BUILD_DIR
make -C $BUILD_DIR install
