#!/bin/bash
set -euo pipefail

# Default env vars
: "${APP_ENV:=production}"

# Exec main command
exec python src/process.py "$@"