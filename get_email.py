import json

import os,sys
import pandas as pd



def read_json(path):
    with open(path,'r',encoding='utf8') as f:
        return json.load(f)


EMAILS_DIR= '/home/mila/c/chris.emezue/scratch/afr/african-digits-recording-sprint-email/emails'

JSON_FILES = [f.name for f in os.scandir(EMAILS_DIR)]


JSON_DICT = [read_json(os.path.join(EMAILS_DIR,f)) for f in JSON_FILES]



df = pd.DataFrame.from_dict(JSON_DICT)
import pdb;pdb.set_trace() 