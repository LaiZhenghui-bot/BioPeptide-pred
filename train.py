# -*- coding: utf-8 -*-
# @Author  : lzh
# @FileName: train.py
# @Software: PyCharm


import os
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from model import base, BiGRU_base,create_umap_model
import time
from test import test_my
import umap

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
import numpy as np
np.random.seed(101)
from pathlib import Path
###
import os
from pathlib import Path
import keras
from keras.optimizers import Adam
from keras.models import model_from_json
# from train import catch
from evaluation import convert_to_labels,eff,Judeff
import pickle
from keras.models import load_model
import keras.backend as K
import numpy as np
from sklearn.metrics import roc_auc_score
import csv
from matplotlib.colors import ListedColormap
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





def train_my(train,test, para, model_num, model_path):

    Path(model_path).mkdir(exist_ok=True)

    ###test[1] = keras.utils.to_categorical(test[1])
    test[0], temp = catch(test[0], test[1])
    temp[temp > 1] = 1
    test[1] = temp
    X_test=test[0]
    y_test=test[1]
    ###
    # data get
    X_train, y_train = train[0], train[1]

    # data and label preprocessing
    y_train = keras.utils.to_categorical(y_train)
    X_train, y_train = catch(X_train, y_train)
    y_train[y_train > 1] = 1

    # disorganize
    index = np.arange(len(y_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    # train
    length = X_train.shape[1]
    out_length = y_train.shape[1]

    t_data = time.localtime(time.time())
    with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
        f.write('data process finished: {}m {}d {}h {}m {}s\n'.format(t_data.tm_mon,t_data.tm_mday,t_data.tm_hour,t_data.tm_min,t_data.tm_sec))
#############################################################################################UMAP可视化

    # x_original = np.copy(X_test).reshape(X_test.shape[0], -1)
    # umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    # embedding_original = umap_model.fit_transform(x_original)
    # plt.figure(figsize=(8, 6))
    # label=test[1]
    # colors = ['red', 'blue']
    # cmap = ListedColormap(colors)
    # plt.scatter(embedding_original[:, 0], embedding_original[:, 1], c=label[:embedding_original.shape[0]], cmap=cmap,
    #             s=5)
    # plt.xlabel('Dimension 1',fontsize=20,fontweight='bold')
    # plt.ylabel('Dimension 2',fontsize=20,fontweight='bold')
    # plt.title('U-map of Human dataset input',fontsize=24,fontweight='bold')
    # plt.savefig('PDCNN_visualization_Original_input.png')
    # plt.show()
    # for counter in range(1, model_num+1):
    #     # get my neural network model
    #     if model_path == 'base':
    #         model = base(length, out_length, para)
    #     elif model_path == 'BiGRU_base':
    #         model= BiGRU_base(length, out_length, para)
    #     else:
    #         print('no model')
    #     # TODO epoch
    #     layer_name = 'output'  # 选择你想要可视化的卷积层
    #     conv_output = model.get_layer(layer_name).output
    #
    #     get_output = K.function([model.input], [conv_output])
    #     model.fit(X_train, y_train, epochs=6, batch_size=64, verbose=2)
    #     intermediate_output = get_output([X_test])[0]
    #     ######
    #     output = intermediate_output.reshape(intermediate_output.shape[0], -1)
    #     reducer = umap.UMAP()
    #     embedding = reducer.fit_transform(output)
    #     ###########
    #     # reducer = umap.UMAP()
    #     # embedding = reducer.fit_transform(intermediate_output)
    #
    #     labels = test[1]
    #     plt.figure(figsize=(8, 6))
    #     colors = ['red', 'blue']
    #     plt.scatter(embedding[:, 0], embedding[:, 1], c=[colors[label] for label in labels], s=5)
    #     plt.xlabel('Dimension 1',fontsize=20,fontweight='bold')
    #     plt.ylabel('Dimension 2',fontsize=20,fontweight='bold')
    #     plt.title('U-MAP of Human dataset output',fontsize=24,fontweight='bold')
    #     # plt.savefig('PDCNN_visualization_Original_input.png')
    #     plt.show()
    #
    #     score = model.predict(X_test)

#####################################################################

        for counter in range(1, model_num + 1):
            # get my neural network model
            if model_path == 'base':
                model = base(length, out_length, para)
            elif model_path == 'BiGRU_base':
                model = BiGRU_base(length, out_length, para)
            else:
                print('no model')
            # TODO epoch

            model.fit(X_train, y_train, epochs=15, batch_size=64, verbose=2)


            score = model.predict(X_test)

#############################################################################

        for i in range(len(score)):
            max_index = np.argmax(score[i])
            for j in range(len(score[i])):
                if j == max_index:
                    score[i][j] = 1
                else:
                    score[i][j] = 0
        preds = convert_to_labels(score)
        AUC = roc_auc_score(y_test, preds)
        TP, FN, FP, TN = eff(y_test, preds)
        # print(TP, FN, FP, TN)
        # AUC = Calauc(y_train, preds)
        SN, SP, ACC, MCC, F1 = Judeff(TP, FN, FP, TN)
        print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
        print("SN: {}, SP: {},F1:{}, ACC: {}, MCC: {}, AUC: {}".format(SN, SP, F1, ACC, MCC, AUC))
        each_model = os.path.join(model_path, 'model' + str(counter) + '.h5')
        model.save(each_model)

        modelname = 'pred'
        date = [modelname, TP, FN, FP, TN, SN, SP, F1, ACC, MCC, AUC]#记录
        csvfile = open('rundate.csv', 'a', encoding='utf-8', newline='')
        writer = csv.writer(csvfile)
        writer.writerow(['TOX','TP', 'FN', 'FP', 'TN','SN', 'SP',"F1",'ACC','MCC','AUC'])#TODO rundate model name
        writer.writerow(date)
        csvfile.close()

        ###
        each_model = os.path.join(model_path, 'model' + str(counter) + '.h5')
        model.save(each_model)

        tt = time.localtime(time.time())
        with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
            f.write('count{}: {}m {}d {}h {}m {}s\n'.format(str(counter),tt.tm_mon,tt.tm_mday,tt.tm_hour,tt.tm_min,tt.tm_sec))




def train_main(train, test, model_num, dir):

    # parameters
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 0.001
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

    # TODO Train test
    train_my(train,test, para, model_num, dir)
    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test start time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))

    # test_my(test, para, model_num, dir)
    # tt = time.localtime(time.time())
    # with open(os.path.join(dir, 'time.txt'), 'a+') as f:
    #     f.write('test finish time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
