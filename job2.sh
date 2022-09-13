#!/bin/bash
#SBATCH --job-name=okwugbe_afro2
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=80GB               # memory (per node)
#SBATCH --time=2-5:50            # time (DD-HH:MM)
#SBATCH --error=/home/mila/c/chris.emezue/okwugbe/slurmerror-%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/okwugbe/slurmoutput-%j.txt

###########cluster information above this line


###load environments
module load python/3
source /home/mila/c/chris.emezue/scratch/okwugbe-afr/bin/activate

cd /home/mila/c/chris.emezue/okwugbe/pytorch_audio_classification
python train.py /home/mila/c/chris.emezue/okwugbe/all_except_ibo_test.csv /home/mila/c/chris.emezue/okwugbe/test_ibo.csv afro_ibo_300_all 



#Igbo - ibo
#Oshiwambo - kua
#Yoruba - yor
#Oromo (although note all of these audios are from female) - gax
#Shona (all male) - sna
#Rundi (all male) - run