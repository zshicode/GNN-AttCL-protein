import torch
import numpy as np
import argparse
import math
import random
import torch.nn.functional as F
from dataGenerate import cazySeq, PadSequence
from torch.utils.data import DataLoader
from model import ConvLstm
from sklearn.metrics import roc_curve,precision_recall_curve,auc
from utils.plot_utils import plotfig

# Set random seed for reproductivity
manualSeed = 999
# manualSeed = random.randint(1, 10000)  # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Set the input parameters of main()
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size used to train the model")
parser.add_argument('-l', "--learning_rate", default=2e-3, type=float, help="learning rate of optimizer")
parser.add_argument('-n', "--num_epochs", default=10, type=int, help="epochs used to train the model")
parser.add_argument('-i', "--input_file", default='train.fasta', help="data path to the training data")
parser.add_argument('-v', "--vec_file", default='glove.txt', help="vector file contains vector of kmers")
parser.add_argument('-t', "--test_file", default='test.fasta', help="data path to the test data")
parser.add_argument('-d', "--device", default="cuda", type=str, help="training engine cuda or cpu")
args = parser.parse_args()

# configure the device
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

def train(model, dataLoader, epoches):
    loss_trace = {'x': [], 'loss': []}
    BatchNums = math.ceil(len(dataLoader.dataset) / dataLoader.batch_size)
    for epoch in range(epoches):
        for i, data in enumerate(dataLoader):
            #seq, seq_lengths, label = data
            seq, label = data
            seq_lengths = 50
            seq, label = seq.to(device), label.to(device)

            output = model(seq, seq_lengths)
            loss = model.criterion(output, label)

            model.optimier.zero_grad()
            # backward
            loss.backward()
            # update the parameters of model
            model.optimier.step()

            if i % 1 == 0:
                print('Epoch[{}/{}], \t Batch[{}/{}], \t Loss:{:.6f}'.format(
                    epoch + 1, epoches, i+1, BatchNums, loss.item()))
                loss_trace['x'].append(epoch + (i+1)/BatchNums)
                loss_trace['loss'].append(loss.item())

    return loss_trace

def test(model, dataLoader):
    print('*'*21 + 'Test Model' + '*'*21)
    model.eval()
    eval_loss = 0.; eval_acc = 0.
    aurocl = []
    auprl = []

    for data in dataLoader:
        seq, label = data
        seq_lengths = 50
        seq, label = seq.to(device), label.to(device)

        output = model(seq, seq_lengths)
        loss = model.criterion(output, label)
        eval_loss += loss.item() * label.size(0)

        # output = F.softmax(output, dim=1)
        output = torch.sigmoid(output)
        # get the predicted labels
        if output.device == 'cpu':
            y_pred = output.detach().numpy().flatten()
            y_true = label.detach().numpy().flatten()
        else:
            y_pred = output.cpu().detach().numpy().flatten()
            y_true = label.cpu().detach().numpy().flatten()
        fpr,tpr,rocth = roc_curve(y_true,y_pred)
        auroc = auc(fpr,tpr)
        precision,recall,prth = precision_recall_curve(y_true,y_pred)
        aupr = auc(recall,precision)
        aurocl.append(auroc)
        auprl.append(aupr)
        _, pred = torch.max(output, 1)
        for i, each_label in enumerate(label):
            if each_label[pred[i]] == 1:
                eval_acc += 1

    eval_loss /= len(dataLoader.dataset)
    eval_acc /= len(dataLoader.dataset)
    aurocl = np.array(aurocl)
    auprl = np.array(auprl)
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    #     eval_loss/len(dataLoader.dataset), eval_acc/len(dataLoader.dataset)))

    # return the loss and accuracy of test dataset
    print("===Final result===")
    print('AUROC= %.4f +- %.4f | AUPR= %.4f +- %.4f' % (aurocl.mean(),aurocl.std(),auprl.mean(),auprl.std()))
    return eval_loss, eval_acc


def main():
    # prepare the training and testing data
    train_data = cazySeq(args.input_file, 6, 2)
    #trainIter = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=PadSequence(), num_workers=2)
    trainIter = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_data = cazySeq(args.test_file, 6, 2)
    #testIter = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=PadSequence(), num_workers=2)
    testIter = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # build and train the model
    model = ConvLstm(args.learning_rate).to(device)
    loss = train(model, trainIter, args.num_epochs)
    # test the model
    eval_loss, eval_acc = test(model, testIter)
    #print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss, eval_acc))

    # plot the training loss curve
    #plt = plotfig()
    #plt.plotCurve([loss['x']], [loss['loss']], ['deepCazy'], 'Training Epoch', 'Loss@CrossEntropy', False)


if __name__ == "__main__":
    main()
