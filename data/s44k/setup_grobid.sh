#!/bin/bash
# This script sets up a running grobid server on port 8070

docker pull grobid/grobid:0.7.1
docker run --rm --gpus all -p 8070:8070 grobid/grobid:0.7.1