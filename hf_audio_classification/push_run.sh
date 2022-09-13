#!/bin/bash

sbatch run_push_to_hub.sh /home/mila/c/chris.emezue/scratch/afr/AFRO_gax_wav.pth  chrisjay/afrospeech-wav2vec-gax
sbatch run_push_to_hub.sh /home/mila/c/chris.emezue/scratch/afr/AFRO_ALL_wav.pth chrisjay/afrospeech-wav2vec-all-6
sbatch run_push_to_hub.sh /home/mila/c/chris.emezue/scratch/afr/AFRO_ibo_wav.pth chrisjay/afrospeech-wav2vec-ibo
sbatch run_push_to_hub.sh /home/mila/c/chris.emezue/scratch/afr/AFRO_kua_wav.pth chrisjay/afrospeech-wav2vec-kua
sbatch run_push_to_hub.sh /home/mila/c/chris.emezue/scratch/afr/AFRO_run_wav.pth chrisjay/afrospeech-wav2vec-run
sbatch run_push_to_hub.sh /home/mila/c/chris.emezue/scratch/afr/AFRO_yor_wav.pth chrisjay/afrospeech-wav2vec-yor
