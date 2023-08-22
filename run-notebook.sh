#!/bin/bash -l

#SBATCH --job-name=ops-jupyter
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --signal=USR2
#SBATCH --exclude=erc-hpc-comp030,erc-hpc-vm0[11-18]
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/users/%u/logs/%j.out

module load python/3.8.12-gcc-9.4.0

# get unused socket per https://unix.stackexchange.com/a/132524
readonly IPADDRESS=$(hostname -I | tr ' ' '\n' | grep '10.211.4.')
readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
cat 1>&2 <<END
1. SSH tunnel from your workstation using the following command:

   Linux and MacOS:
   ssh -NL 8888:${HOSTNAME}:${PORT} ${USER}@hpc.create.kcl.ac.uk

   Windows:
   ssh -m hmac-sha2-512 -NL 8888:${HOSTNAME}:${PORT} ${USER}@hpc.create.kcl.ac.uk

   and point your web browser to http://localhost:8888/lab?token=<add the token from the jupyter output below>

When done using the notebook, terminate the job by
issuing the following command on the login node:

      scancel -f ${SLURM_JOB_ID}

END

source /users/${USER}/.bashrc
source activate /scratch/users/${USER}/conda/meta-learning-lines
cd /scratch/users/k21036268/meta-learned-lines
jupyter-lab --port=${PORT} --ip=${IPADDRESS} --no-browser

printf 'notebook exited' 1>&2
printf 'notebook exited' 1>&2
