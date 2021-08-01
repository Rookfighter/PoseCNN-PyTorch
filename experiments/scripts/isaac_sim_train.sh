#!/bin/bash

set -x
set -e

./tools/train_net.py \
  --network posecnn \
  --dataset isaac_sim \
  --cfg experiments/cfgs/isaac_sim.yml \
  --solver sgd \
  --epochs 16
