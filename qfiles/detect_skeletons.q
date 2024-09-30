#! /bin/bash
#SBATCH --job-name=analysis
#SBATCH --partition=gpu
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --output=/dev/null
#SBATCH --mem=32G
module load miniconda3
conda activate DT_C
module load cudnn/8.9.7.29-12-e6mhbo5

FILESSOURCE=$1

echo "Username: " `whoami`
FSOURCE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $FILESSOURCE)
echo ${FSOURCE}
eval $FSOURCE

exit 0

