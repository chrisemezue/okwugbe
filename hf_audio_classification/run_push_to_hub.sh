#!/bin/bash
#SBATCH --job-name=push_to_hub
#SBATCH --mem=80GB               # memory (per node)
#SBATCH --time=2-5:50            # time (DD-HH:MM)
#SBATCH --error=/home/mila/c/chris.emezue/okwugbe/slurmerror_push_to_hub-%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/okwugbe/slurmoutput_push_to_hub-%j.txt

###########cluster information above this line


###load environments
module load python/3
source /home/mila/c/chris.emezue/scratch/okwugbe-afr/bin/activate

cd /home/mila/c/chris.emezue/okwugbe/hf_audio_classification

python push_model.py $1 $2
