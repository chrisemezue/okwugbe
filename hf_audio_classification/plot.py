import seaborn as sns
import json
import pandas as pd




def get_data(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

d_types = ['ALL','ibo','gax','kua','run','sna','yor']

loss_stylr = '/home/mila/c/chris.emezue/okwugbe/hf_audio_classification/loss_AFRO_{}_wav.json'
acc_style = '/home/mila/c/chris.emezue/okwugbe/hf_audio_classification/val_acc_AFRO_{}_wav.json'

def plot_(type='loss'):
    style = loss_stylr if type=='loss' else acc_style
    lang_array=[]
    value = []
    ids=[]
    
    for d in d_types:
        data = get_data(style.format(d))
        value.extend(data)
        lang_array.extend([d for i in data])
        ids.extend([i for i in range(len(data))])
    return value,lang_array,ids


#Val acc plot
v,l,ids_ = plot_('Val_Acc')

df = pd.DataFrame({'Val_Acc':v,'Language':l,'Step':ids_})

ax = sns.relplot(
    data=df, x="Step", y="Val_Acc",
    col="Language", 
    kind="line",col_wrap=3
)
ax.set_titles('Validation accuracy')
ax.savefig('/home/mila/c/chris.emezue/okwugbe/hf_audio_classification/val_accuracy_plot.png')




#Loss plot
v,l,ids_ = plot_('loss')

df = pd.DataFrame({'Loss':v,'Language':l,'Step':ids_})

ax = sns.relplot(
    data=df, x="Step", y="Loss",
    col="Language", 
    kind="line",col_wrap=3
)
ax.set_titles('Training Loss')
ax.savefig('/home/mila/c/chris.emezue/okwugbe/hf_audio_classification/loss_main_plot.png')
