#!/bin/bash -l

#SBATCH --mem=4096
#SBATCH --job-name=meta-learning-lines
#SBATCH --partition=gpu
#SBATCH --exclude=erc-hpc-comp040,erc-hpc-comp030,erc-hpc-comp033,erc-hpc-vm0[11-18]
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/users/%u/logs/%j.out
#SBATCH --signal=SIGUSR1@90

nvidia-debugdump -l
hostname

while getopts ":m:o:i:l:s:f:" option; do
  case $option in
    m)
      modelType="$OPTARG"
      ;;
    o)
      outerLR="$OPTARG"
      ;;
    i)
      innerLR="$OPTARG"
      ;;
    l)
      outputLR="$OPTARG"
      ;;
    s)
      steps="$OPTARG"
      ;;
    f)
      lengthTasks="$OPTARG"
      ;;
    *)
      exit 1
      ;;
  esac
done

module --ignore_cache load rstudio
module --ignore_cache load python/3.8.12-gcc-9.4.0
module --ignore_cache load cuda/11.7.0-gcc-10.3.0
source /users/${USER}/.bashrc
source activate /scratch/users/${USER}/conda/meta-learning-lines
cd /scratch/users/k21036268/meta-learned-lines
srun python main.py --model $modelType --outerLR $outerLR --innerLR $innerLR --outputLR $outputLR --steps $steps --batchSize 8 --warmupSteps 100 --kShot 8 --kValShot 8 --numTasks 2000000 --lengthTasks $lengthTasks