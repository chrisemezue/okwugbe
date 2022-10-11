import os, sys
import numpy as np
import random
import torch
from torch import nn
from transformers import AutoFeatureExtractor
from IPython.display import Audio, display
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from okwugbedataset import OkwugbeDataset
from utils import get_dataset, create_data_loader,train,evaluate



model_checkpoint = "facebook/wav2vec2-base"
batch_size = 32
num_labels = 10

metric = load_metric("accuracy")


label2id, id2label = dict(), dict()
labels = ['0','1','2','3','4','5','6','7','8','9']

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


"""

To get a sense of what the commands sound like, the following snippet will render 
some audio examples picked randomly from the dataset. 




for _ in range(5):
    rand_idx = random.randint(0, len(dataset["train"])-1)
    example = dataset["train"][rand_idx]
    audio = example["audio"]

    print(f'Label: {id2label[str(example["label"])]}')
    print(f'Shape: {audio["array"].shape}, sampling rate: {audio["sampling_rate"]}')
    display(Audio(audio["array"], rate=audio["sampling_rate"]))
    print()
"""


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}")

SAMPLE_RATE = 48000
FILE_NAME = sys.argv[1]
OUTPUT_PATH = f'/home/mila/c/chris.emezue/scratch/afr/{FILE_NAME}'
os.makedirs(OUTPUT_PATH,exist_ok=True)
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
EPOCHS = 50


#Preprocessing the data
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 2.0  # seconds


usd,valid_dataset,test_dataset=  get_dataset(feature_extractor,
                                                16000,
                                                device,
                                                train_path,
                                                test_path)
# Create dataloader
train_dataloader = create_data_loader(usd, batch_size)
valid_dataloader = create_data_loader(valid_dataset, batch_size)
test_dataloader = create_data_loader(test_dataset, batch_size)

"""
To apply this function on all utterances in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command.
"""



# construct model and assign it to device
model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint, 
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
).to(device)



# initialise loss funtion + optimiser
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)


# Let's try using Trainer
training_args = TrainingArguments(output_dir=OUTPUT_PATH,evaluation_strategy="epoch", push_to_hub=True,
                                    hub_model_id =f"chrisjay/{FILE_NAME.lower()}",
                                    report_to ='none',
                                    optim='adamw_hf',
                                    save_total_limit=2,
                                    logging_strategy ='steps',
                                    logging_steps = 50, 
                                    logging_nan_inf_filter = True,
                                    num_train_epochs = EPOCHS,
                                    learning_rate  = LEARNING_RATE
    )



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=usd,
    eval_dataset=valid_dataset,
    compute_metrics=evaluate,
)

trainer.train()
trainer.push_to_hub()

'''
# train model
train(model, train_dataloader,valid_dataloader, loss_fn, optimiser, device, EPOCHS,SAVE_PATH,LOSS_JSON_FILE,ACC_JSON_FILE,EVAL_STEP)
if test_dataloader is not None:
    test_acc = evaluate(model, test_dataloader,device)
    print(f"Test accuracy is {test_acc}")
else:
    print(f"No test dataset was provided! So not performing evaluation on test.")
'''
print('ALL DONE')



'''

model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-afros-speech",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)




def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)    



trainer.train()

'''