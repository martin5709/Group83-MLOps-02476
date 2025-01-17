### HOW TO USE                                                                          ###
### Start by initializing sweep yourself from the CL using                              ###
### wandb sweep configs/wand_sweep.yaml                                                 ###
### Remember to change the name of wand_sweep.yaml, if you have made a new config file  ###
### Copy the sweep id, it should appear in the terminal in a line like                  ###
### wandb: Creating sweep with ID: ya38ax08                                             ###
###                                ^^^^^^^^                                             ###
###                               sweepd_id                                             ###
### To submit batch job run                                                             ###
### bsub -env "all,SWEEP_ID=sweep_id" < wandb_sweep.sh                                  ###
###                         ^^^^^^^^                                                    ###
###                      your new sweep_id                                              ###



#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Weights_and_Biases_Sweep
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 2:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=2GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s203768@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o hpc_logs/%J.out
#BSUB -e hpc_logs/%J.err
# -- end of LSF options --

#nvidia-smi
# Load the cuda module
module load cuda/11.6

source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlops

if [ -z "$SWEEP_ID" ]; then
  echo "Error: SWEEP_ID environment variable is not set!"
  exit 1
fi

wandb agent group83-MLOps-02476/group83-MLOps-02476/$SWEEP_ID