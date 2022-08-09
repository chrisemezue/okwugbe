import matplotlib.pyplot as plt




t_loss, v_loss = [],[]
filename = "slurmoutput.txt"
with open(filename,'r') as f:
    lines = f.readlines()

for i in range(len(lines)):
    if lines[i].strip().startswith('Loss'):
        train_loss = float(lines[i+1].strip().split('cur:')[1].strip(')'))
        valid_loss = float(lines[i+2].strip().split('cur:')[1].strip(')'))

        t_loss.append(train_loss)
        v_loss.append(valid_loss)

indices = [i for i in range(len(t_loss))]
fig,ax = plt.subplots()

ax.plot(indices,t_loss)
ax.plot(indices,v_loss)
ax.set(xlabel='Train step', ylabel='Loss',title='Train and Valid loss')
plt.savefig('loss_plot2.png')
print('ALL DONE')

