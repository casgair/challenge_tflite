#!/usr/bin/env bash

# Go to root dir
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1

# Test
cd ./build/macos || exit 1
ctest --verbose
