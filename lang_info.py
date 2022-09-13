import pandas as pd




lang_dict={}

for i in range(0,10):
    lang_dict.update({str(i):[]})

df = pd.read_csv('/home/mila/c/chris.emezue/okwugbe/stats_afro_dataset.csv') 

u_langs = df['lang'].unique().tolist()
l_numbers=[]
for u_lang in u_langs:
    df_lang = df[df['lang']==u_lang]
    for i in range(0,10):
        df_number = df_lang[df_lang['transcript']==i]
        lang_dict[str(i)].append(len(df_number))

save_dict = {'languages':u_langs}
save_dict.update(lang_dict)
final_df = pd.DataFrame(save_dict)
final_df.to_csv('lang_count_per_digit.csv',index=False)

