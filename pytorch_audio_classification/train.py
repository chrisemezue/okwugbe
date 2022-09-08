import os
import sys
import json
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from urbansounddataset import OkwugbeDataset
from cnn import CNNNetwork


def dump_json(thing,file):
    with open(file,'w+',encoding="utf8") as f:
        json.dump(thing,f)



def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    _ = iter(data_loader).next()
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    loss_item = loss.item()

    return loss_item

def train(model, data_loader,valid_dataloader,test_dataloader, loss_fn, optimiser, device, epochs,SAVE_PATH,LOSS_JSON_FILE,ACC_JSON_FILE):
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
            acc = evaluate(model,valid_dataloader)
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


def evaluate(model, data_loader):
    model.eval()
    with torch.no_grad():
        acc=[]
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)

            # calculate accuracy
            prediction = model(input)
            predicted_index = prediction.argmax(1)

            train_acc = torch.sum(predicted_index == target).cpu().item()
            final_train_acc = train_acc/input.shape[0]
            acc.append(final_train_acc)
    
    final_acc = sum(acc)/len(acc) 
    
    return final_acc

def get_dataset(mel_spectogram,device,train_path,test_path):
    usd = OkwugbeDataset(mel_spectrogram,
                            SAMPLE_RATE,
                            train_path,
                            test_path,
                            'train',
                            device)
    valid_dataset = OkwugbeDataset(mel_spectrogram,
                            SAMPLE_RATE,
                            train_path,
                            test_path,
                            'valid',
                            device)
    test_dataset = OkwugbeDataset(mel_spectrogram,
                            SAMPLE_RATE,
                            train_path,
                            test_path,
                            'test',
                            device)

    return usd,valid_dataset,test_dataset



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    BATCH_SIZE = 128
    EPOCHS = 500
    EVAL_STEP=5
    LEARNING_RATE = 0.001
    SAMPLE_RATE = 48000

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    FILE_NAME = sys.argv[3]
    SAVE_PATH = f'/home/mila/c/chris.emezue/scratch/afr/{FILE_NAME}.pth'

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    LOSS_JSON_FILE = f'/home/mila/c/chris.emezue/okwugbe/pytorch_audio_classification/loss_{FILE_NAME}.json'
    ACC_JSON_FILE = f'/home/mila/c/chris.emezue/okwugbe/pytorch_audio_classification/val_acc_{FILE_NAME}.json'



    usd,valid_dataset,test_dataset=  get_dataset(mel_spectrogram,
                                                device,
                                                train_path,
                                                test_path)
    # Create dataloader
    train_dataloader = create_data_loader(usd, BATCH_SIZE)
    valid_dataloader = create_data_loader(valid_dataset, BATCH_SIZE)
    test_dataloader = create_data_loader(test_dataset, BATCH_SIZE)

    # construct model and assign it to device
    model = CNNNetwork().to(device)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(model, train_dataloader,valid_dataloader,test_dataloader, loss_fn, optimiser, device, EPOCHS,SAVE_PATH,LOSS_JSON_FILE,ACC_JSON_FILE)
    test_acc = evaluate(model, test_dataloader)
    print(f"Test accuracy is {test_acc}")

    print('ALL DONE')
