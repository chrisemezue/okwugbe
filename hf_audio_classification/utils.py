import os
import sys
import json
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from okwugbedataset import OkwugbeDataset
#from urbansounddataset import OkwugbeDataset
#from cnn import CNNNetwork


def dump_json(thing,file):
    with open(file,'w+',encoding="utf8") as f:
        json.dump(thing,f)



def get_weighted_sampler(train_data_):
    # https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a
    train_data = train_data_.original_data

    if train_data is None:
        return None

    unique_langs = train_data['lang_code'].unique().tolist() 
    
    if len(unique_langs)<=1:
        # Only one language, do not do weighted sampling
        return None   
    lang_count = [train_data[train_data['lang_code']==k].shape[0] for k in unique_langs]
    lang_weight = [1/c for c in lang_count]

    weight_lang_dict = {l:w for l,w in zip(unique_langs,lang_weight)}

    def get_lang_weight(lang):
        return weight_lang_dict[lang]

    train_data['lang_weight'] = train_data['lang_code'].apply(lambda x: get_lang_weight(x))    

    language_weights_for_data =  train_data['lang_weight'].values.tolist()

    weighted_sampler = WeightedRandomSampler(
    weights=language_weights_for_data,
    num_samples=len(language_weights_for_data),
    replacement=True
        )

    return weighted_sampler    


def create_data_loader(train_data, batch_size):
    if train_data is not None:
        weighted_sampler = get_weighted_sampler(train_data)

        if weighted_sampler is not None:
            print('='*30)
            print('Using weighted sampler')
            print('='*30)

            train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=False,sampler=weighted_sampler)
        else:    
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
    else:
        train_dataloader = None
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    _ = iter(data_loader).next()
    for input, target in data_loader:

        input, target = input.input_values[0].squeeze().to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction.logits, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    loss_item = loss.item()

    return loss_item

def train(model, data_loader,valid_dataloader,loss_fn, optimiser, device, epochs,SAVE_PATH,LOSS_JSON_FILE,ACC_JSON_FILE,EVAL_STEP):
    best_acc = 0
    for i in range(epochs):
        print(f"Epoch {i}")
        loss_item = train_single_epoch(model, data_loader, loss_fn, optimiser, device)

        if os.path.exists(LOSS_JSON_FILE):
            with open(LOSS_JSON_FILE,'r') as f:
                loss_array = json.load(f)
        else:
            loss_array=[]

        loss_array.append(loss_item)            
        print(f"loss: {loss_item}")

        dump_json(loss_array,LOSS_JSON_FILE)


        if i%EVAL_STEP==0:
            acc = evaluate(model,valid_dataloader,device)
            print(f'Validation accuracy: {acc}')

            if os.path.exists(ACC_JSON_FILE):
                with open(ACC_JSON_FILE,'r') as f:
                    acc_array = json.load(f)
            else:
                acc_array=[]

            acc_array.append(acc)            

            dump_json(acc_array,ACC_JSON_FILE)
            
            if acc > best_acc: 
                # save model
                torch.save(model.state_dict(), SAVE_PATH )
                print(f"Trained feed forward net saved at {SAVE_PATH}")
                best_acc= acc
        print("---------------------------")
    print("Finished training")


def evaluate(model, data_loader,device):
    model.eval()
    with torch.no_grad():
        acc=[]
        for input, target in data_loader:


            input, target = input.input_values[0].squeeze().to(device), target.to(device)


            # calculate accuracy
            prediction = model(input).logits
            predicted_index = prediction.argmax(1)

            train_acc = torch.sum(predicted_index == target).cpu().item()
            final_train_acc = train_acc/input.shape[0]
            acc.append(final_train_acc)
    
    final_acc = sum(acc)/len(acc) 
    
    return final_acc

def get_dataset(mel_spectogram,SAMPLE_RATE,device,train_path,test_path):
    usd = OkwugbeDataset(mel_spectogram,
                            SAMPLE_RATE,
                            train_path,
                            test_path,
                            'train',
                            device)
    valid_dataset = OkwugbeDataset(mel_spectogram,
                            SAMPLE_RATE,
                            train_path,
                            test_path,
                            'valid',
                            device)
    if test_path is not None:
        test_dataset = OkwugbeDataset(mel_spectogram,
                            SAMPLE_RATE,
                            train_path,
                            test_path,
                            'test',
                            device)
    else: 
        test_dataset = None

    return usd,valid_dataset,test_dataset


