#!/bin/bash
mkdir synthetic_data

python simulator.py
for i in {1..60}
do 
    python simulator.py "$i"
done