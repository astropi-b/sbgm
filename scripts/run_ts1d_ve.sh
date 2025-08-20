#!/bin/bash
# Train and sample a VE-SDE model on synthetic 1D time series.
python -m sbgm.cli.main --config configs/ts1d_ve.yaml --train