#!/bin/bash
module unload python3/3.9.7
export CUDA_HOME=/opt/apps/cuda/12.0/
export TRITON_CACHE_DIR="/scratch/09697/luosong/cache"
source /work/09697/luosong/frontera/anaconda3/bin/activate pathphi
torchrun --nproc_per_node=3 finetuning.py \
                            --batch_size 12 \
                            --batch_size_per_gpu 4 \
                            --use_flash_attention \
                            --use_qlora \
                            --bf16
