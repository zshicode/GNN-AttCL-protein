import numpy as np
import pandas as pd

li = pd.read_csv('drug_list_zhang.csv')
size = len(li)
train = pd.read_csv('ZhangDDI_train.csv')
val = pd.read_csv('ZhangDDI_valid.csv')
test = pd.read_csv('ZhangDDI_test.csv')
df = pd.concat([train,val,test])

a = np.zeros((size,size))
for idx,row in df.iterrows():
    # or use i = li.query("drugbank_id==row['drugbank_id_1']").index[0]
    i = li.loc[li['drugbank_id']==row['drugbank_id_1']].index[0]
    j = li.loc[li['drugbank_id']==row['drugbank_id_2']].index[0]
    a[i,j] = row['label']

np.savetxt('interact_mat.txt',a,'%.1f')