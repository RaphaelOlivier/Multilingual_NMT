#!/bin/sh
python code/vocab.py \
python code/run.py \
    train \

python code/run.py decode
