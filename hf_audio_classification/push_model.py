import os, sys
import numpy as np
import torch
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification




PATH = sys.argv[1]
save_name = sys.argv[2]

model_checkpoint = "facebook/wav2vec2-base"
batch_size = 32
num_labels = 10


label2id, id2label = dict(), dict()
labels = ['0','1','2','3','4','5','6','7','8','9']

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}")


#Preprocessing the data
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 2.0  # seconds


# construct model and assign it to device
model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint, 
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
).to(device)



model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))

model.push_to_hub(f"chrisjay/{save_name}")
feature_extractor.push_to_hub(f"chrisjay/{save_name}")
print('ALL DONE')

#python push_model.py /home/mila/c/chris.emezue/scratch/afr/AFRO_sna_wav.pth chrisjay/afrospeech-wav2vec-sna