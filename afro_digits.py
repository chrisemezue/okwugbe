
import os
import json
import pandas as pd
from okwugbe.train_eval import Train_Okwugbe



audio_path=[]
audio_transcript=[]
audio_lang=[]
audio_lang_code=[]
audio_gender = []
audio_age = []
audio_country = []
audio_accent = []


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


audio_dict = {
    'audio_path':audio_path,
    'transcript':audio_transcript,
    'lang':audio_lang,
    'lang_code':audio_lang_code,
    'gender':audio_gender,
    'age':audio_age,
    'country':audio_country,
    'accent':audio_accent
    }

df = pd.DataFrame(audio_dict)
                  
df.to_csv('all_afro_dataset.csv',index=False)

#Split to train, test and valid

# Split by language

df_lang = df[df['lang_code']=='ibo']
df_lang['transcript'] = df_lang['transcript'].astype(str)
train=df_lang.sample(frac=0.8,random_state=200) #random state is a seed value
test=df_lang.drop(train.index)



train.to_csv('train_ibo.csv',index=False)

test.to_csv('test_ibo.csv',index=False)

'''
train_path is a CSV file with two columns: 
        first column is the path to your audio files
        second column is the transcripts

test_path is a CSV file with two columns: 
        first column is the path to your audio files
        second column is the transcripts

characters_set is a TXT file for the alphabets of the language, including their diacritics.
        Each letter of the alphabet goes on each line.
        See https://github.com/edaiofficial/okwugbe/blob/main/fon/characters_set.txt for an example with Fon
'''

"""Load the OkwuGbe model"""


train_path = 'train_ibo.csv'
#train_path='all_afro_dataset.csv'
test_path = 'test_ibo.csv'
characters_set = 'characters.txt'
model_path='/home/mila/c/chris.emezue/scratch/okwugbe_igbo_numbers'

train_asr = Train_Okwugbe(train_path, test_path, characters_set,model_path=model_path)

"""Train the model"""

train_asr.run()