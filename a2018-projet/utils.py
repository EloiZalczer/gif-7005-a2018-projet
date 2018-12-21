#!/usr/bin/python

import numpy as np
from sklearn import metrics

def compute_stats(output, labels):

    stats = []

    # print("Labels : ", labels)
    print("Labels shape : ", labels.shape)
    print("Output shape : ", output.shape)

    for c in range(labels.shape[1]):
        # Average precision
        avg_precision = metrics.average_precision_score(
            labels[:, c], output[:, c], average=None)

        # AUC
        auc = metrics.roc_auc_score(labels[:, c], output[:, c], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            labels[:, c], output[:, c])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(labels[:, c], output[:, c])

        save_every_steps = 1000  # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc}
        stats.append(dict)

    return stats

def compute_mean_stats(output, labels):
    stats = compute_stats(output, labels)

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])

    return mAP, mAUC