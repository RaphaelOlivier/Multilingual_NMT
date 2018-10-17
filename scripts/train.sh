#!/bin/sh

work_dir="results"

python code/nmt.py \
    train \

python code/nmt.py decode
