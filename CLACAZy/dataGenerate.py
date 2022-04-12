import torch
import re
#import gensim
import biovec
import numpy as np
from Bio import SeqIO
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class PadSequence:
    '''Pad the sequences of different lengths, and this class is modified slightly
    from https://www.codefull.org/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/
    '''
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # See the __getitem__ method in Dataset class for details to create dataset
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        # pad_sequence(): Arguments: list[Tensor]
        sequences_padded = pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = [len(x) for x in sequences]
        # Get the labels of sorted batch. Note: map() and list(Tensors) cannot be accepted by torch.Tensor()
        labels = torch.Tensor([x[1] for x in sorted_batch])
        return sequences_padded, lengths, labels


class cazySeq(Dataset):
    """
    Define the dataset class from cazy sequence data (.fasta file).
    """
    def __init__(self, seq_file, kmer, stride, size=50):
        # map the cazy family to class number
        self.cate2label = {'GH':0, 'GT':1, 'PL':2, 'CE':3, 'AA':4, 'CBM':5}
        self.cateNums = len(self.cate2label)
        # read the .fasta file to get sequences and their corresponded labels
        self.seq, self.seqLabel = self.readSeq(seq_file)
        # the kmer and stride parameters used to convert sequences to tensors
        self.kmer = kmer; self.stride = stride
        # load the word2vec model using gensim
        # self.vectors = gensim.models.KeyedVectors.load_word2vec_format(vec_file)
        # self.embedDimension = self.vectors.vectors.shape[1]
        self.embedDimension = self.kmer*size
        self.pv = biovec.models.ProtVec(
            seq_file,corpus_fname=seq_file+'out.txt',n=self.kmer,size=size)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        # Address the label first
        label = np.zeros(self.cateNums, dtype=np.float32)
        for each_label in self.seqLabel[index]:
            label[each_label] = 1.
        # Address the data
        seq_tensor = self.tensorFromSeq(self.seq[index])
        return seq_tensor, label

    def readSeq(self, seq_file):
        # read the .fasta file to get sequences and their corresponded labels
        seqList = []; seqLabelList = []
        matchPtn = r'^GH|^GT|^PL|^CE|^AA|^CBM'
        for seq_record in SeqIO.parse(seq_file, "fasta"):
            seqList.append( str(seq_record.seq).upper() )
            seqId = seq_record.id.split('|')[1:]
            matchedList = []
            for eachId in seqId:
                matchedList.extend( re.findall(matchPtn, eachId) )
            matchedSet = set(matchedList)
            seqLabelList.append([self.cate2label[each_cat] for each_cat in matchedSet])
        return seqList, seqLabelList

    def tensorFromSeq(self, seq):
        # convert a sequence to a tensor
        '''
        seq_tensor = []
        for i in range(0, len(seq) - self.kmer + 1, self.stride):
            if seq[i:(i+self.kmer)] in self.vectors.vocab:
                seq_tensor.append(self.vectors[seq[i:(i+self.kmer)]])
                #seq_tensor.append(self.pv[seq[i:(i+self.kmer)]])
            else:
                seq_tensor.append(np.zeros(self.embedDimension, dtype=np.float32))
        seq_tensor = torch.Tensor(seq_tensor)
        '''
        seqa = torch.tensor(self.pv.to_vecs(seq))#3 * size = 3*50
        seqa = torch.cat((seqa,seqa),dim=0) #6*50
        seq_tensor = torch.cat((seqa,seqa),dim=0) #12*50
        return seq_tensor
