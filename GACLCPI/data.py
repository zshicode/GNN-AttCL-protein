import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
import re
import csv
import torch
from rdkit import Chem

def getName(l,p):
    ldf = pd.DataFrame(columns=['id','compound','smiles'])
    i,j = 0,0
    for d in l.keys():
        ldf = ldf.append({'id':i,'compound':d,'smiles':l[d]},ignore_index=True)
        i += 1
    ldf.to_csv('data/'+dataset+'_drug_name.csv',index=False)
    pdf = pd.DataFrame(columns=['id','protein','sequence'])
    with open('data/' + dataset + '_protein.fasta', 'w') as f: 
        for d in p.keys():
            pdf = pdf.append({'id':j,'protein':d,'sequence':p[d]},ignore_index=True)
            f.write('>'+d+'\n')
            f.write(p[d]+'\n')
            j += 1
    pdf.to_csv('data/'+dataset+'_protein_name.csv',index=False)

def getAffinityMatrix(a):  
    rows, cols = np.where(np.isnan(a)==True)
    for i in range(len(rows)):
        a[rows[i],cols[i]] = 0.0
    np.savetxt('data/'+dataset+'_affinity_matrix.txt',a,'%.5f',delimiter=',')
    # a.shape (num_drug, num_target)

datasets = ['davis','kiba']
for dataset in datasets:
    print('===Processing '+dataset+'===')
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    #train_fold = [ee for e in train_fold for ee in e ]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    folds = train_fold + [valid_fold]
    valid_ids = [5,4,3,2,1]
    valid_folds = [folds[vid] for vid in valid_ids]
    train_folds = []
    for i in range(5):
        temp = []
        for j in range(6):
            if j != valid_ids[i]:
                temp += folds[j]
        train_folds.append(temp)
    
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        #lg = ligands[d]
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]

    affinity = np.asarray(affinity)
    opts = ['train','test']
    for i in range(5):
        train_fold = train_folds[i]
        valid_fold = valid_folds[i]
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity)==False)  
            if opt=='train':
                rows,cols = rows[train_fold], cols[train_fold]
            elif opt=='test':
                rows,cols = rows[valid_fold], cols[valid_fold]
                
            if i == 0:
                #generating standard data
                print('generating standard data')
                with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
                    f.write('compound_iso_smiles,target_sequence,affinity,protein_id\n')
                    for pair_ind in range(len(rows)):
                        ls = []
                        ls += [ drugs[rows[pair_ind]]  ]
                        ls += [ prots[cols[pair_ind]]  ]
                        ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                        ls += [ cols[pair_ind] ]
                        f.write(','.join(map(str,ls)) + '\n')
                
                #generating cold data
                print('generating cold data')
                if opt=='train':
                    with open('data/' + dataset + '_cold' + '.csv', 'w') as f:
                        f.write('compound_iso_smiles,target_sequence,affinity,protein_id,drug_id\n')
                        for pair_ind in range(len(rows)):
                            ls = []
                            ls += [ drugs[rows[pair_ind]]  ]
                            ls += [ prots[cols[pair_ind]]  ]
                            ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                            ls += [ cols[pair_ind] ]
                            ls += [ rows[pair_ind] ]
                            f.write(','.join(map(str,ls)) + '\n') 
                else:
                    with open('data/' + dataset + '_cold' + '.csv', 'a') as f:
                        for pair_ind in range(len(rows)):
                            ls = []
                            ls += [ drugs[rows[pair_ind]]  ]
                            ls += [ prots[cols[pair_ind]]  ]
                            ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                            ls += [ cols[pair_ind] ]
                            ls += [ rows[pair_ind] ]
                            f.write(','.join(map(str,ls)) + '\n') 
                      
            #5-fold validation data
            print('generating 5-fold validation data')
            with open('data/' + dataset + '/' + dataset + '_' + opt + '_fold' + str(i) + '.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity,protein_id\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [ drugs[rows[pair_ind]]  ]
                    ls += [ prots[cols[pair_ind]]  ]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    ls += [ cols[pair_ind] ]
                    f.write(','.join(map(str,ls)) + '\n')       
    
    getName(ligands,proteins)
    getAffinityMatrix(affinity)
