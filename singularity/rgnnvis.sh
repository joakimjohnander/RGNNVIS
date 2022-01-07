#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --time 72:00:00
#SBATCH --mem-per-gpu 64G
#SBATCH --output /workspaces/%u/2022-rgnnvis/logs/slurm-%j-run.out
#

singularity exec --bind /workspaces/$USER:/workspace \
	    --bind /data/ytvis:/my_data/ytvis \
	    --bind /data/openimages:/my_data/openimages_image \
	    --bind /data/openimages/Annotations:/my_data/openimages_anno \
	    --pwd /workspace/RGNNVIS/ \
	    --env PYTHONPATH=/workspace/RGNNVIS/:/workspace/RGNNVIS/rgnnvis/ \
	    /workspaces/$USER/RGNNVIS/singularity/pytorch21_09.sif \
	    python3 -u $@

#
#EOF
