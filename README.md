# Graph neural networks and attention-based CNN-LSTM for protein classification

This paper proposes three models for protein classification. Firstly, this paper proposes a Multi-label CAZyme classification model using CNN-LSTM with Attention mechanism.  Secondly, this paper proposes a variational graph autoencoder based subspace learning model for protein graph classification. Thirdly, this paper proposes  Graph Isomorphism Networks and Attention-based CNN-LSTM for compound-protein interactions prediction. 

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

The data include three datasets
- `GACLCPI/dataset/human` includes the human dataset from https://github.com/masashitsubaki/CPI_prediction (Tsubaki et al. 2019) or https://github.com/XuanLin1991/GraphCPI. (Quan et al. 2019) This dataset models compound-protein interaction prediction as binary classification task.
- `GACLCPI/data/davis` and `GACLCPI/data/kiba` include the DAVIS and KIBA datasets from https://github.com/hkmztrk/DeepDTA/tree/master/data (Ozturk et al. 2018) or https://github.com/yuanweining/FusionDTA. (Yuan et al. 2022) These datasets model compound-protein interaction (drug-target affinity) prediction as regression task.

Both of classification task and regression task modeling this problem, can be seen as representation learning for compound-protein (drug-target, ligand-protein) pairs. So these tasks in different applied perspective can be unified into same framework (Liu et al. 2021; Yang et al. 2022).

The code (on human dataset for classification task) can be run through
```
cd GACLCPI/code
python preprocess_data.py
python run_training.py
```

The script `GACLCPI/data.py` preprocesses the DAVIS and KIBA datasets for regression task. 
- The DAVIS and KIBA datasets are transformed into the list of compound-protein pairs (in CSV files). 
- The lists of compounds (including name and SMILE of each compound) and proteins (including name and sequence of each protein) are created (in CSV files). Protein names and sequences are also formatted as FASTA files.
- The affinity values are storaged as matrices. The matrix shape is `CompoundNum*ProteinNum`. This step can formulate the drug-target affinity regression task as matrix factorization for biological association prediction. 

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

### References

Gao et al. Graph u-nets, in ICML, 2019.

Liu et al. Persistent spectral hypergraph based machine learning (PSH-ML) for protein-ligand binding affinity prediction, Briefings in Bioinformatics, 22(5):bbab127, 2021.

Ozturk et al. DeepDTA: deep drug–target binding affinity prediction, Bioinformatics, 34(17):i821-i829, 2018.

Quan et al. GraphCPI: Graph Neural Representation Learning for Compound-Protein Interaction, in IEEE BIBM, 2019.

Tsubaki et al. Compound–protein interaction prediction with end-to-end learning of neural networks for graphs and sequences, Bioinformatics, 35(2):309-318, 2019.

Yang et al. MGraphDTA: deep multiscale graph neural network for explainable drug–target binding affinity prediction, Chemical Science, 13(3):816-833, 2022.

Yuan et al. FusionDTA: attention-based feature polymerizer and knowledge distillation for drug-target binding affinity prediction, Briefings in Bioinformatics, 23(1):bbab506, 2022.

Zhang et al. dbCAN2: a meta server for automated carbohydrate-active enzyme annotation. Nucleic Acids Research, 46:W95-W101, 2018.