#!/usr/bin/env bash

initialize() {
	if test "$BASH" = "" || "$BASH" -uc 'a=();true "${a[@]}"' 2>/dev/null; then
		# Bash 4.4, Zsh
		set -euo pipefail
		shopt -s nullglob globstar
	else
		# Bash 4.3 and older chokes on empty arrays with set -u.
		set -eo pipefail
	fi
}

initialize_destination_dir() {
	local DEST_DIR="$1"
	mkdir -p "$DEST_DIR"
}

print_destination_dir() {
	local DEST_DIR="$1"
	echo ""
	echo "Libs are in $DEST_DIR"
}
