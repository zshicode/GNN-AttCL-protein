# Graph neural networks and attention-based CNN-LSTM for protein classification

This repo proposes three models for protein classification.
- Multi-label CAZyme classification model using CNN-LSTM with Attention mechanism.  
- variational graph autoencoder based model for protein graph classification. 
- Graph Isomorphism Networks and Attention-based CNN-LSTM for compound-protein interactions prediction. 

Besides, this repo collects and collates the benchmark datasets for three problems mentioned above. Hence, the usage for evaluation by benchmark datasets can be more conveniently. Datasets include: 
- CAZyme classification
- Enzyme protein graph classification
- Compound-protein interactions prediction
- Drug-target affinities prediction 
- Drug-drug interactions prediction

## Usage

### CLACAZy

The dbCAN2 (https://bcb.unl.edu/dbCAN2/index.php) database (Zhang et al. 2018) collected 1,066,327 CAZymes until July 31, 2018. The data can be downloaded via the website of dbCAN2 (https://bcb.unl.edu/dbCAN2/download/CAZyDB.07312018.fa). 

We select 4683 CAZymes from all 1,066,327 CAZymes, to construct the dataset. 3759 CAZymes are divided into training set, while the other 924 CAZymes are divided into testing set. See `train.fasta` and `test.fasta` .

The code can be run through
```
cd CLACAZy
python main.py
```

Hyper parameter setting can refer to our code repository.

### VSPool

Data formed for VSPool experiments can be downloaded from this link: (https://github.com/HongyangGao/Graph-U-Nets/tree/master/data) (Gao et al. 2019).

After downloading the data, the code can be run through
```
cd VSPool
```
Type
```
./run_GNN.sh DATA FOLD GPU
```
to run on dataset using fold number (1-10).

You can run
```
./run_GNN.sh DD 0 0
```
to run on DD dataset with 10-fold cross validation on GPU #0.

### GACLCPI

#### Compound-protein interactions prediction

`GACLCPI/dataset/human` includes the human dataset from https://github.com/masashitsubaki/CPI_prediction (Tsubaki et al. 2019) or https://github.com/XuanLin1991/GraphCPI. (Quan et al. 2019) This dataset models compound-protein interaction prediction as binary classification task.

The code (on human dataset for classification task) can be run through
```
cd GACLCPI/code
python preprocess_data.py
python run_training.py
```

The code implements GCN, GAT and GIN for compound embedding (default is GIN), in function `gnn` of class `CompoundProteinInteractionPrediction`.
```python
class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        # ...

    def gnn(self, xs, A, layer, model=GNNMODEL):
        if model=='GCN':
            # graph convolution
        elif model=='GAT':
            # graph attention
        else:
            # default is GIN
```

#### Drug-target affinities prediction

`GACLCPI/data/davis` and `GACLCPI/data/kiba` include the DAVIS and KIBA datasets from https://github.com/hetong007/SimBoost (He et al. 2017) (by R language) or https://github.com/hkmztrk/DeepDTA/tree/master/data (Ozturk et al. 2018) or https://github.com/yuanweining/FusionDTA. (Yuan et al. 2022) These datasets model compound-protein interaction (drug-target affinity) prediction as regression task.

Both of classification task and regression task modeling this problem, can be seen as representation learning for compound-protein (drug-target, ligand-protein) pairs. Each pair denotes a sample. So these tasks in different applied perspective can be unified into same framework.

The script `GACLCPI/data.py` preprocesses the DAVIS and KIBA datasets for regression task. 
- The DAVIS and KIBA datasets are transformed into the list of compound-protein pairs (in CSV files). 
- The lists of compounds (including name and SMILE of each compound) and proteins (including name and sequence of each protein) are created (in CSV files). Protein names and sequences are also formatted as FASTA files.
- The affinity values are storaged as matrices. The matrix shape is `CompoundNum*ProteinNum`. This step can formulate the drug-target affinity regression task as matrix factorization for biological association prediction. 

#### Drug-drug interactions prediction

`GACLCPI/ddi/miracle_ddi.py` transforms datasets in MIRACLE model (Wang et al. 2021) (https://github.com/isjakewong/MIRACLE)  for DDI. The data folder is (https://github.com/isjakewong/MIRACLE/tree/main/MIRACLE/datachem). The labels are transformed to be storaged as matrices. The matrix shape is `DrugNum*DrugNum`. This step can formulate the Drug-drug interactions prediction task from drug-drug pair classification (such that a pair denotes a sample) to matrix factorization. 

Chen et al. (2021) adopted MUFFIN model (https://github.com/xzenglab/MUFFIN) that uses DRKG dataset (https://github.com/gnn4dr/DRKG/) for Drug-drug interactions (DDI) binary classification and multi-class classification prediction. DRKG is Drug Repurposing Knowledge Graph for COVID-19.

`GACLCPI/ddi/drkg_ddi.py` transforms DRKG dataset for DDI. The data folder is (https://github.com/xzenglab/MUFFIN/tree/main/data/DRKG). The labels are transformed to be storaged as matrices. The matrix shape is `DrugNum*DrugNum`.

### Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:
```
numpy
pytorch
sklearn
biopython
biovec
rdkit
```

## References

Chen et al. MUFFIN: multi-scale feature fusion for drug–drug interaction prediction, Bioinformatics, 37(17):2651-2658, 2021.

Gao et al. Graph u-nets, in ICML, 2019.

He et al. SimBoost: a read-across approach for predicting drug–target binding affinities using gradient boosting machines, Journal of Cheminformatics, 9:24, 2017.

Ozturk et al. DeepDTA: deep drug–target binding affinity prediction, Bioinformatics, 34(17):i821-i829, 2018.

Quan et al. GraphCPI: Graph Neural Representation Learning for Compound-Protein Interaction, in IEEE BIBM, 2019.

Tsubaki et al. Compound–protein interaction prediction with end-to-end learning of neural networks for graphs and sequences, Bioinformatics, 35(2):309-318, 2019.

Wang et al. Multi-view Graph Contrastive Representation Learning for Drug-Drug Interaction Prediction, in WWW, 2021.

Yuan et al. FusionDTA: attention-based feature polymerizer and knowledge distillation for drug-target binding affinity prediction, Briefings in Bioinformatics, 23(1):bbab506, 2022.

Zhang et al. dbCAN2: a meta server for automated carbohydrate-active enzyme annotation. Nucleic Acids Research, 46:W95-W101, 2018.
