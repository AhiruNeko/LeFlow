#!/bin/bash

# Train the H=10 joint path/action flow with the downloaded Fast-LeWM model.
export WORLD_MODEL_TYPE=fast-lewm
export WORLD_MODEL_CHECKPOINT=/root/projects/pusht/fast-lewm/pusht/Fast-lewm_pusht_object.ckpt
exec bash shell/train_pusht_h10.sh
