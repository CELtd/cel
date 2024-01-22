#!/bin/bash

N_DEVICES=8

XLA_FLAGS="--xla_force_host_platform_device_count=${N_DEVICES}" \
    python baseline_bug_monitor.py \
    --starboard-token /home/kiran/code/cel/auth/spacescope_auth.json \
    --num-warmup-mcmc 10000 \
    --num-samples-mcmc 2000 \
    --seasonality-mcmc 2000 \
    --num-chains-mcmc ${N_DEVICES}

