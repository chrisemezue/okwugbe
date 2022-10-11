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

# Running for each language of interest 
bash job_wav2vec.sh afrospeech-wav2vec-all-6 /home/mila/c/chris.emezue/okwugbe/all_interesred_6_audiodata.csv
bash job_wav2vec.sh afrospeech-wav2vec-ibo /home/mila/c/chris.emezue/okwugbe/lang_specific/igbo_ibo_audio_data.csv
bash job_wav2vec.sh afrospeech-wav2vec-gax /home/mila/c/chris.emezue/okwugbe/lang_specific/oromo_gax_audio_data.csv
bash job_wav2vec.sh afrospeech-wav2vec-kua /home/mila/c/chris.emezue/okwugbe/lang_specific/oshiwambo_kua_audio_data.csv
bash job_wav2vec.sh afrospeech-wav2vec-run /home/mila/c/chris.emezue/okwugbe/lang_specific/rundi_run_audio_data.csv
bash job_wav2vec.sh afrospeech-wav2vec-sna /home/mila/c/chris.emezue/okwugbe/lang_specific/shona_sna_audio_data.csv
bash job_wav2vec.sh afrospeech-wav2vec-yor /home/mila/c/chris.emezue/okwugbe/lang_specific/yoruba_yor_audio_data.csv
