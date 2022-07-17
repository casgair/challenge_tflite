#!/usr/bin/env bash

print_usage() {
	echo "Usage: $0 -v VERSION [-d]" 1>&2
	echo "    -d   build debug version"
	exit 1
}

# Go to root dir
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1

# Initialize
source ./bin/shared.sh
initialize

# Parse options
BUILD_TYPE="Release"

while getopts "d" opt; do
	case "${opt}" in
	d)
		BUILD_TYPE="Debug"
		;;
	*)
		print_usage
		;;
	esac
done

# Destination dir
DEST_DIR="$(pwd)/build/macos"
initialize_destination_dir "$DEST_DIR"

# Configure
cd "$DEST_DIR" || exit 1
cmake ../.. \
	-DCMAKE_BUILD_TYPE="$BUILD_TYPE"
echo ""

# Build
make all -j"$(getconf _NPROCESSORS_ONLN)"
print_destination_dir "$DEST_DIR"
