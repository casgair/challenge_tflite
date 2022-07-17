#!/usr/bin/env bash

# Go to root dir
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1

# Test
cd ./build/linux || exit 1
ctest --verbose
