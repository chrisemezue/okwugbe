#!/bin/bash

# Running for each language of interest 
sbatch job_wav2vec.sh AFRO_ibo_wav /home/mila/c/chris.emezue/okwugbe/lang_specific/igbo_ibo_audio_data.csv
sbatch job_wav2vec.sh AFRO_gax_wav /home/mila/c/chris.emezue/okwugbe/lang_specific/oromo_gax_audio_data.csv
sbatch job_wav2vec.sh AFRO_kua_wav /home/mila/c/chris.emezue/okwugbe/lang_specific/oshiwambo_kua_audio_data.csv
sbatch job_wav2vec.sh AFRO_run_wav /home/mila/c/chris.emezue/okwugbe/lang_specific/rundi_run_audio_data.csv
sbatch job_wav2vec.sh AFRO_sna_wav /home/mila/c/chris.emezue/okwugbe/lang_specific/shona_sna_audio_data.csv
sbatch job_wav2vec.sh AFRO_yor_wav /home/mila/c/chris.emezue/okwugbe/lang_specific/yoruba_yor_audio_data.csv
sbatch job_wav2vec.sh AFRO_ALL_wav /home/mila/c/chris.emezue/okwugbe/all_interesred_6_audiodata.csv
