#!/bin/bash
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-4              # job array with index values 0, 1, 2, 3
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=jylin@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

PYTHON_SCRIPT_NAME=$(basename "$SLURM_JOB_NAME" .slurm).py
DATA_DIR='/home/jylin/thesis/final/data'
RES_DIR='/home/jylin/thesis/final/resources'

module purge
module load anaconda3/2020.11
conda activate thesis

python $PYTHON_SCRIPT_NAME \
    --data_dir $DATA_DIR \
    --res_dir $RES_DIR \
    --max_hidden_size 5 \
    --task_id $SLURM_ARRAY_TASK_ID
