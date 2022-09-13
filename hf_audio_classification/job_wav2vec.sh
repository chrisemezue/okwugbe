#!/bin/bash
#SBATCH --job-name=okwugbe_afro_wav2vec
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=80GB               # memory (per node)
#SBATCH --time=2-5:50            # time (DD-HH:MM)
#SBATCH --error=/home/mila/c/chris.emezue/okwugbe/slurmerror-%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/okwugbe/slurmoutput-%j.txt

###########cluster information above this line


###load environments
module load python/3
source /home/mila/c/chris.emezue/scratch/okwugbe-afr/bin/activate

cd /home/mila/c/chris.emezue/okwugbe/hf_audio_classification

#python train.py afro_ibo_300_all /home/mila/c/chris.emezue/okwugbe/lang_specific/igbo_ibo_audio_data.csv /home/mila/c/chris.emezue/okwugbe/test_ibo.csv  

python train.py $1 $2 $3



