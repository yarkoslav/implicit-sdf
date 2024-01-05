#!/bin/bash

mkdir -p test_out
export PATH="/usr/local/cuda-12.0/bin:/home/ubuntu-system/miniconda3/envs/tinycudann/bin:/home/ubuntu-system/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
num_cycles=50

for ((i = 0; i < num_cycles; i++)); do
    python main_sdf.py test_task_meshes/$i.obj --workspace test_out/$i --fp16 --tcnn
    # ls -lah test_out/$i/checkpoints/ngp.pth.tar
done
