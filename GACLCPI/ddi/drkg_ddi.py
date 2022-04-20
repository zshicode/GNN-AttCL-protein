import numpy as np
'''
li = np.loadtxt('drugname_smiles.txt')
size = len(li)
'''
size = 2322
df = np.loadtxt('DDI_pos_neg.txt',dtype=int)
d = np.loadtxt('multi_ddi_sift.txt',dtype=int)

a = np.zeros((size,size))
b = np.zeros((size,size))
for i in range(len(df)):
    a[df[i,0],df[i,1]] = df[i,2]

for i in range(len(d)):
    b[d[i,0],d[i,1]] = d[i,2]

np.savetxt('interact_mat.txt',a,'%d')
np.savetxt('multi_interact_mat.txt',b,'%d')