#!/bin/bash

act -j build-and-test --pull=false -P self-hosted=ghcr.io/catthehacker/ubuntu:act-latest -P gpu=ghcr.io/catthehacker/ubuntu:act-latest -P linux=ghcr.io/catthehacker/ubuntu:act-latest