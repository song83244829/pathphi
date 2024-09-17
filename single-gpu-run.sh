#!/bin/bash
BATCH_SIZE_PER_GPU=2

module unload python3/3.9.7
export CUDA_HOME=/opt/apps/cuda/12.0/
export TRITON_CACHE_DIR="/scratch/09697/luosong/cache"
source /work/09697/luosong/frontera/anaconda3/bin/activate pathphi
torchrun --nproc_per_node=3 temp.py \
                            --data_dir /scratch/09697/luosong/databases/quilt_1M \
                            --batch_size $(($BATCH_SIZE_PER_GPU * 3)) \
                            --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
                            --num_train_epochs 2 \
                            --use_flash_attention \
                            --use_qlora \
                            --bf16
