""" classify_stats.py

    Purpose: Contains a dataclass to handle classifying/training statistics
    Author: Milan Marocchi
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix


class RunningBinaryConfusionMatrix:
    def __init__(self):
        self.base_stats = defaultdict(int)
        self.loss = 0

        self.aliases = dict(
            loss='Loss',
            acc='Acc',
            tpr='Recall(Sens)',
            tnr='Specif',
            fpr='FPR',
            ppv='Prec',
            npv='NPV',
            f1p='F1+',
            f1n='F1-',
            acc_mu='Mean Acc',
            j='j',
            mcc='MCC',
            qi='qi'
        )

    def update(self, y_true, y_pred, loss):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        self.base_stats['tn'] += tn
        self.base_stats['fp'] += fp
        self.base_stats['fn'] += fn
        self.base_stats['tp'] += tp

        self.loss += loss

    def total(self):
        return sum([self.base_stats[stat] for stat in self.base_stats])

    def tp(self):
        return self.base_stats['tp']

    def tn(self):
        return self.base_stats['tn']

    def fp(self):
        return self.base_stats['fp']

    def fn(self):
        return self.base_stats['fn']

    def get_stats(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            stats = dict(
                loss=self.loss/self.total(),
                tpr=self.tp()/sum([self.tp(), self.fn()]),
                tnr=self.tn()/sum([self.tn(), self.fp()]),
                fpr=self.fp()/sum([self.fp(), self.tn()]),
                ppv=self.tp()/sum([self.tp(), self.fp()]),
                npv=self.tn()/sum([self.tn(), self.fn()]),
                acc=sum([self.tp(), self.tn()])/self.total(),
            )

            stats.update(dict(  # type: ignore
                acc_mu=np.mean([stats['tpr'], stats['tnr']]),
                qi=np.sqrt(np.prod(stats['tpr']*stats['tnr'])),
                j=sum([stats['tpr'], stats['tnr'], -1]),
                f1p=(2*stats['ppv']*stats['tpr'])/(stats['ppv']+stats['tpr']),
                f1n=(2*stats['npv']*stats['tnr'])/(stats['npv']+stats['tnr']),
            ))

            mcc_numer = (self.tp() * self.tn()) - (self.fp() * self.fn())
            mcc_denom = (
                (self.tp() + self.fp()) *
                (self.tp() + self.fn()) *
                (self.tn() + self.fp()) *
                (self.tn() + self.fn())
            ) ** 0.5

            if mcc_denom <= 1e-8:
                mcc_denom = 1

            mcc = mcc_numer / mcc_denom
            stats.update(dict(  # type: ignore
                mcc=mcc,
            ))

        return stats

    def display_stats(self, aliases=True):
        if aliases is True:
            aliases = self.aliases
        else:
            aliases = {}

        stats = self.get_stats()

        for a in stats:
            if a not in aliases:
                aliases[a] = a

        formatted_stats = ', '.join(
            f'{aliases[stat]}: {stats[stat]:.3f}' for stat in aliases)
        return formatted_stats
