#!/bin/bash

#SBATCH --job-name=Finch_hier_70B
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH -w augi4
#SBATCH -p batch
#SBATCH -t 240:00:00
#SBATCH -o out_log/%N_%x_%j.out
#SBTACH -e %x_%j.err
source /data/minkuk/init.sh
conda activate LLM
python finch-llama_hier.py --LLM_size 70 --LLM_use True --target_domain vitt
python finch-llama_hier.py --LLM_size 70 --LLM_use True --target_domain yc2