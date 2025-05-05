#!/bin/bash
# Unset PYTHONPATH to avoid any local Python settings
unset PYTHONPATH
# Unset any conda-related environment variables
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PYTHON_EXE

path=/home/martinez-garcia/discrepancy_vae/
cd "${path}"

pwd=$(pwd)
export HOME="$(pwd)"
export PYTHONPATH='/home/martinez-garcia/discrepancy_vae/'
export WANDB_API_KEY="740950e76ec925e646d291f15ceb318879365c62"

/home/martinez-garcia/micromamba/envs/discrepancy_vae/bin/python "$@"