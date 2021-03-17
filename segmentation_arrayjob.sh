#!/bin/sh
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

CONDA_ENV_NAME=mlp
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS - change line below if loc different
repo_home=/home/${USER}/git/MLP-project
src_path=${repo_home}/datasets

# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_HOME}/dronedeploy/datasets
mkdir -p ${dest_path}  # make it if required

rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

experiment_text_file=$1
exp_name=${experiment_text_file::-14}
mkdir -p ${dest_path}/output/${exp_name}
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"

echo "Moving output data back to DFS"

src_path=${dest_path}/output/${exp_name}
dest_path=${repo_home}/${exp_name}
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


