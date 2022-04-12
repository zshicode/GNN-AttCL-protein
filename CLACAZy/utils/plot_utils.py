import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

class plotfig:

    def __init__(self, dpi = 600, xylabel_fontsize = 15):
        self.colorstyle = ['b--', 'g--', 'r--']
        self.colors = cycle(['navy', 'deeppink', 'aqua', 'darkorange', 'cornflowerblue'])
        self.dpi = dpi
        self.xylabel_fontsize = xylabel_fontsize

    def plotBar(self, barData, n_levels, in_groupTag, xlabelTag, isShow):
        plt.figure(figsize=(4, 3.2))
        ax = plt.gca()
        index = np.arange(n_levels)
        bar_width = 0.35
        opacity = 0.4

        for i, color in zip(range(len(barData)), self.colors):
            plt.bar(index + i * bar_width, barData[i], bar_width, alpha=opacity,
                    label = in_groupTag[i], color=color)

        plt.xlabel('Dataset', fontsize = self.xylabel_fontsize)
        plt.ylabel('ROC-AUC', fontsize = self.xylabel_fontsize)
        plt.xticks(index + bar_width / 2)
        ax.set_xticklabels(xlabelTag)
        plt.legend(loc = 'best', edgecolor = 'k', fancybox = False)

        plt.tight_layout()
        plt.savefig('Barplot of %s.pdf' % '_'.join(in_groupTag), dpi = self.dpi)
        if isShow:
            plt.show()
        plt.close()

    def plotAuc(self, fpr, tpr, auc_score, labelTag, isShow):
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'

        plt.figure(figsize=(4, 3.2))
        ax = plt.gca()
        # assert len(default_style) >= len(labelTag), 'too many curves to plot' not necessary using circle

        for i, color in zip(range(len(labelTag)), self.colors):
            plt.plot(fpr[i], tpr[i], color = color, linestyle = ':',
                     lw = 1, label = labelTag[i] + ' Auc = %0.3f' % auc_score[i])
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        plt.grid(True, linestyle = ':')
        plt.legend(loc = 'lower right', edgecolor = 'k', fancybox = False)
        plt.xlabel('False Positive Rate', fontsize = self.xylabel_fontsize)
        plt.ylabel('True Positive Rate',  fontsize = self.xylabel_fontsize)

        plt.tight_layout()
        plt.savefig('Auc of %s.pdf' % '_'.join(labelTag), dpi = self.dpi)
        if isShow:
            plt.show()
        plt.close()

    def plotCurve(self, x, y, labelTag, xlabel, ylabel, isShow):
        plt.figure(figsize=(4, 3.2))
        ax = plt.gca()

        for i, color in zip(range(len(labelTag)), self.colors):
            plt.plot(x[i], y[i], color = color, linestyle = ':',
                     lw = 1, label = labelTag[i])
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        plt.grid(True, linestyle = ':')
        plt.legend(loc = 'best', edgecolor = 'k', fancybox = False)
        plt.xlabel(xlabel, fontsize = self.xylabel_fontsize)
        plt.ylabel(ylabel, fontsize = self.xylabel_fontsize)

        plt.tight_layout()
        plt.savefig('Curve of %s.pdf' % '_'.join(labelTag), dpi = self.dpi)
        if isShow:
            plt.show()
        plt.close()
