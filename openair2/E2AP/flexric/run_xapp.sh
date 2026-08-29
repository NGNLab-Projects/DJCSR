#!/bin/bash

# Force the OS to load PyTorch's CPU library first to avoid Intel oneMKL conflicts
export LD_PRELOAD=/home/ngnlab/.local/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so

# Set the FlexRIC Python SDK path
export PYTHONPATH=/home/ngnlab/openairinterface5g/openair2/E2AP/flexric/build/src/xApp/swig/

# Run the xApp
python3 /home/ngnlab/openairinterface5g/openair2/E2AP/flexric/examples/xApp/python3/isac_xapp.py
