#!/bin/bash
set -euo pipefail

# Default env vars
: "${APP_ENV:=production}"

# Exec main command
echo "Starting Perfect-Shot watcher..."
exec python src/process.py