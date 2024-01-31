#!/bin/bash

N_DEVICES=16

XLA_FLAGS="--xla_force_host_platform_device_count=${N_DEVICES}" \
    python baseline_bug_monitor.py \
    --mcmc-train-len-days 90 \
    --starboard-token /home/kiran/code/cel/auth/spacescope_auth.json \
    --num-warmup-mcmc 10000 \
    --num-samples-mcmc 4000 \
    --seasonality-mcmc 2000 \
    --num-chains-mcmc ${N_DEVICES}

