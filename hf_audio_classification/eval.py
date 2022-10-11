import os, sys
import json
import numpy as np
import random
import torch
import seaborn as sns
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix,f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from datasets import load_dataset, load_metric
from okwugbedataset_eval import OkwugbeDataset
from utils import get_dataset, create_data_loader,train,evaluate


CM_PATH = '/home/mila/c/chris.emezue/okwugbe/hf_audio_classification/metrics_valid'

model_checkpoint = "facebook/wav2vec2-base"
batch_size = 64
num_labels = 10

metric = load_metric("accuracy")


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

SAMPLE_RATE = 48000
FILE_NAME = sys.argv[1]
SAVE_PATH = f'/home/mila/c/chris.emezue/scratch/afr/{FILE_NAME}.pth'
EVAL_STEP=1
train_path = sys.argv[2]
try:
    test_path = sys.argv[3]
except IndexError:
    # The test path was not provided.
    test_path = None     

LOSS_JSON_FILE = f'/home/mila/c/chris.emezue/okwugbe/hf_audio_classification/loss_{FILE_NAME}.json'
ACC_JSON_FILE = f'/home/mila/c/chris.emezue/okwugbe/hf_audio_classification/val_acc_{FILE_NAME}.json'

LEARNING_RATE = 3e-5
EPOCHS = 150

#Preprocessing the data

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 2.0  # seconds



def plot_bar(value,name,x_name,y_name,title):
    fig, ax = plt.subplots(tight_layout=True)

    ax.set(xlabel=x_name, ylabel=y_name,title=title)

    ax.barh(name, value)
   
  
    return ax.figure 
    
# get distribution of train dataset
df_bar = pd.read_csv(train_path)

digits_dict = Counter(df_bar['transcript'].values.tolist())

digits_name_for_language = list(digits_dict.keys())
digits_count_for_language = [digits_dict[k] for k in digits_name_for_language]
plt_digits = plot_bar(digits_count_for_language,digits_name_for_language,'Number of audio samples',"Digit",f"Audio samples over digits ({FILE_NAME.upper()}) ")

plt_digits.savefig(os.path.join(CM_PATH,f'digits-bar-plot-for-{FILE_NAME}.png'))

usd,valid_dataset,test_dataset=  get_dataset(feature_extractor,
                                                16000,
                                                device,
                                                train_path,
                                                test_path)
# Create dataloader
#train_dataloader = create_data_loader(usd, batch_size)
valid_dataloader = create_data_loader(valid_dataset, batch_size)
#test_dataloader = create_data_loader(test_dataset, batch_size)

"""
To apply this function on all utterances in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command.
"""

def evaluate_full(model, data_loader,device,type_):
    if type_ not in ['valid','test']:
        raise Exception(f"`type_` must be either `test` or `valid`!")


    # Get acc, f1 and confusion matrix
    model.eval()
    with torch.no_grad():
        acc=[]

        preds, targets = [],[]
        for input, target in data_loader:


            input, target = input.to(device), target.to(device)


            # calculate accuracy
            prediction = model(input).logits
            predicted_index = prediction.argmax(1)

            preds.extend(predicted_index.cpu().numpy().tolist())
            targets.extend(target.cpu().numpy().tolist())

            train_acc = torch.sum(predicted_index == target).cpu().item()
            final_train_acc = train_acc/input.shape[0]
            acc.append(final_train_acc)
    
    final_acc = sum(acc)/len(acc) 
    f1_scores = f1_score(targets, preds, average='weighted').tolist() 


    # Creating  a confusion matrix,which compares the y_test and y_pred
    cm = confusion_matrix(targets, preds)

    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(cm)
                        
    #Plotting the confusion matrix
    plt.figure()
    sns.heatmap(cm_df, annot=True)
    plt.title(f'Confusion Matrix ({FILE_NAME.upper()}) ({type_.upper()})')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig(os.path.join(CM_PATH,f'{FILE_NAME.lower()}_confusion_matrix_{type_.upper()}.png'))

    data = {'acc':final_acc,'f1':f1_scores}
    METRICS_FILE = os.path.join(CM_PATH,f'{FILE_NAME.lower()}_METRICS_{type_.upper()}.json')
    with open(METRICS_FILE,'w+') as file_:
        json.dump(data,file_)

    return final_acc



# construct model and assign it to device

model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint, 
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
).to(device)

if os.path.exists(SAVE_PATH):
    print('Using pretrained model.......................')
    model_saved = torch.load(SAVE_PATH)
    model.load_state_dict(model_saved)


# initialise loss funtion + optimiser
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)


# train model
#train(model, train_dataloader,valid_dataloader, loss_fn, optimiser, device, EPOCHS,SAVE_PATH,LOSS_JSON_FILE,ACC_JSON_FILE,EVAL_STEP,feature_extractor)

_ = evaluate_full(model, valid_dataloader,device,'valid')

print('ALL DONE')