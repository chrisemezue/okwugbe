import matplotlib.pyplot as plt
import json


LOSS_JSON_PATH = '/home/mila/c/chris.emezue/okwugbe/pytorch_audio_classification/loss_afro_ibo_300_all.json'
VAL_ACC_JSON_PATH = '/home/mila/c/chris.emezue/okwugbe/pytorch_audio_classification/val_acc_afro_ibo_300_all.json'


with open(LOSS_JSON_PATH,'r') as f:
    loss_json = json.load(f)



with open(VAL_ACC_JSON_PATH,'r') as f:
    val_acc_json = json.load(f)



def plot_(x,y,title,x_label,y_label,fig_save):
    fig,ax = plt.subplots()

    ax.plot(x,y)
    ax.set(xlabel=x_label, ylabel=y_label,title=title)
    plt.savefig(fig_save)
    print('ALL DONE')

plot_([i for i in range(len(loss_json))],loss_json,'Training Loss','Step','Loss','loss_plot_500_all.png')

plot_([i for i in range(len(val_acc_json))],val_acc_json,'Validation accuracy','Step','Accuracy','val_acc_plot_500_all.png')#plot val acc

