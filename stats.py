
import os
import json
import torchaudio
import pandas as pd

'''
audio_path=[]
audio_transcript=[]
audio_lang=[]
audio_lang_code=[]
audio_gender = []
audio_age = []
audio_country = []
audio_accent = []
audio_length = []

audio_dirs = '/home/mila/c/chris.emezue/scratch/afr/data'
audio_folders = [os.path.join(audio_dirs,f) for f in os.listdir(audio_dirs)]


for folder in audio_folders:
  audio_path_ = os.path.join(folder,'audio.wav') #need to check this exists
  audio_metadata = os.path.join(folder,'metadata.jsonl') #need to check this exists
  if os.path.isfile(audio_metadata):
    if os.path.isfile(audio_path_):
      with open(audio_metadata,'r') as file:
        metadata = json.load(file)
      
      audio_path.append(audio_path_)
      audio_transcript.append(str(metadata['number']))
      audio_lang.append(metadata['language_name'])
      audio_lang_code.append(metadata['language_id'])
      audio_age.append(metadata['age'])
      audio_country.append(metadata['country'])
      audio_accent.append(metadata['accent'])
      audio_gender.append(metadata['gender'])
      waveform, sample_rate = torchaudio.load(audio_path_)
      duration = waveform.shape[1]/sample_rate
      audio_length.append(duration)


audio_dict = {
    'audio_path':audio_path,
    'transcript':audio_transcript,
    'lang':audio_lang,
    'lang_code':audio_lang_code,
    'gender':audio_gender,
    'age':audio_age,
    'country':audio_country,
    'accent':audio_accent,
    'duration':audio_length
    }

df = pd.DataFrame(audio_dict)


df.to_csv('stats_afro_dataset.csv',index=False)
print('ALL DONE')

'''

df = pd.read_csv('stats_afro_dataset.csv')
import seaborn as sns
import matplotlib.pyplot as plt 

fig,ax = plt.subplots()
#ax = sns.histplot(data=df, x="duration")
ax = sns.violinplot(x=df["duration"])
plt.title('Distribution of audio duration in dataset')
plt.savefig('audio_duration_v.png')
print('ALL DONE')