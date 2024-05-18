# -*- coding: utf-8 -*-
# @Author  : lzh
# @FileName: test.py
# @Software: PyCharm


import os
from pathlib import Path
import keras
from keras.optimizers import Adam
from keras.models import model_from_json
# from train import catch
from evaluation import convert_to_labels,eff,Judeff
import pickle
from keras.models import load_model
import numpy as np
import json
import csv
from sklearn.metrics import roc_auc_score
def catch(data, label):
    # preprocessing label and data
    l = len(data)
    chongfu = 0
    for i in range(l):
        ll = len(data)
        idx = []
        each = data[i]
        j = i + 1
        bo = False
        while j < ll:
            if (data[j] == each).all():
                label[i] += label[j]
                idx.append(j)
                bo = True
            j += 1
        t = [i] + idx
        if bo:
            # print(t)
            chongfu += 1
            # print(data[t[0]])
            # print(data[t[1]])
        data = np.delete(data, idx, axis=0)
        label = np.delete(label, idx, axis=0)

        if i == len(data)-1:
            break
    print('total number of the same data: ', chongfu)

    return data, label


def predict(X_test, y_test, thred, para, weights, jsonFiles, h5_model, dir):

    # with open('test_true_label.pkl', 'wb') as f:
    #     pickle.dump(y_test, f)

    adam = Adam(lr=para['learning_rate']) # adam optimizer
    for ii in range(0, len(weights)):
        # 1.loading weight and structure (model)

        # json_file = open('BiGRU_base/' + jsonFiles[i], 'r')
        # model_json = json_file.read()
        # json_file.close()
        # load_my_model = model_from_json(model_json)
        # load_my_model.load_weights('BiGRU_base/' + weights[i])
        # load_my_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        h5_model_path = os.path.join(dir, h5_model[ii])
        load_my_model = load_model(h5_model_path)


        print("Prediction is in progress")

        # 2.predict
        score = load_my_model.predict(X_test)

        "========================================"
        for i in range(len(score)):
            max_index = np.argmax(score[i])
            for j in range(len(score[i])):
                if j == max_index:
                    score[i][j] = 1
                else:
                    score[i][j] = 0

        # print(preds)
        AUC = roc_auc_score(y_test, score)
        preds = convert_to_labels(score)
        y_test = convert_to_labels(y_test)
        TP, FN, FP, TN = eff(y_test, preds)
        SN, SP, ACC, MCC, F1 = Judeff(TP, FN, FP, TN)
        print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
        print("SN: {}, SP: {}, F1:{}, ACC: {}, MCC: {}, AUC: {}".format(SN, SP, F1, ACC, MCC, AUC))
        modelname = 'TOX'   #TODO MODEL NAME
        date = [modelname, TP, FN, FP, TN, SN, SP, F1, ACC, MCC, AUC]
        csvfile = open('test_rundate.csv', 'a', encoding='utf-8', newline='')
        writer = csv.writer(csvfile)
        # writer.writerow(['Model','TP', 'FN', 'FP', 'TN','SN', 'SP',"F1",'ACC','MCC','AUC'])
        writer.writerow(date)
        csvfile.close()
        # 3.evaluation
        if ii == 0:
            score_label = score
        else:
            score_label += score

    score_label = score_label / len(h5_model)
    # data saving
    with open(os.path.join(dir, 'MLBP_prediction_prob.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    # getting prediction label
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < thred: score_label[i][j] = 0
            else: score_label[i][j] = 1

    # data saving
    with open(os.path.join(dir, 'MLBP_prediction_label.pkl'), 'wb') as f:
        pickle.dump(score_label, f)





def test_my(test, para, model_num, dir):
    # step1: preprocessing
    test[1] = keras.utils.to_categorical(test[1])
    test[0], temp = catch(test[0], test[1])
    temp[temp > 1] = 1
    test[1] = temp

    # weight and json
    weights = []
    jsonFiles = []
    h5_model = []
    for i in range(1, model_num+1):
        weights.append('model{}.hdf5'.format(str(i)))
        # jsonFiles.append('model{}.json'.format(str(i)))
        h5_model.append('model{}.h5'.format(str(i)))



    # step2:predict
    predict(test[0], test[1], test[2], para, weights, jsonFiles, h5_model, dir)