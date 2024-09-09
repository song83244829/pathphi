#!/bin/bash
#SBATCH -J pathphi           # Job name
#SBATCH -o pathphi.o%j       # Name of stdout output file
#SBATCH -e pathphi.e%j       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 4               # Total # of nodes 
#SBATCH -n 4              # Total # of mpi tasks
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A MCB23087       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=luo.song@bcm.edu

# make sure -N and -n match with ibrun

GPU_NUM=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

ibrun -np $GPU_NUM ./run_finetuning.sh $MASTER_ADDR $GPU_NUM