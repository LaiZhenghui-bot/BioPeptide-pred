import math
import numpy as np

def convert_to_labels(y_hat):
    n, m = y_hat.shape
    labels=[0]*n
    # 遍历矩阵的每个元素
    for i in range(n):
        for j in range(m):
            if y_hat[i, j] == 1:
                labels[i]=j
    return labels
def eff(labels, preds):

    TP, FN, FP, TN = 0, 0, 0, 0

    for idx,label in enumerate(labels):
        if label == 1:
            if label == preds[idx]:
                TP += 1
            else: FN += 1
        elif label == preds[idx]:
            TN += 1
        else: FP += 1

    return TP, FN, FP, TN

def Judeff(TP, FN, FP, TN):

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    precision = TP / (TP + FP)
    recall =SN
    ACC = (TP + TN) / (TP + FN + FP + TN)
    MCC = (TP * TN - FP * FN) / (math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    f1_score = 2 * precision * recall / (precision + recall)
    return SN, SP, ACC, MCC,f1_score