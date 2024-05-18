# -*- coding: utf-8 -*-
# @Author  : lzh
# @FileName: demo.py
# @Software: PyCharm


import os
import time
import numpy as np
from pathlib import Path
import random
dir = 'BiGRU_base'
Path(dir).mkdir(exist_ok=True)
t = time.localtime(time.time())
with open(os.path.join(dir, 'time.txt'), 'w') as f:
    f.write('start time: {}m {}d {}h {}m {}s'.format(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    f.write('\n')
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import random

def GetSourceData(root, dir, lb):
    seqs = []
    print('\n')
    print('now is ', dir)
    file = '{}.txt'.format(dir)
    file_path = os.path.join(root, dir, file)

    # with open(file_path) as f:
    #     for each in f:
    #         if each == '\n' or each[0] == '>':
    #             continue
    #         else:
    #             seqs.append(each.rstrip())

    with open(file_path) as f:
        all_seqs = f.readlines()  # 读取文件中的所有行
        num_seqs = len(all_seqs)  # 获取文件中数据的总数量
        # TODO DATA RATIO.

        k=1
        num_to_extract = int(num_seqs * k)  
        random.shuffle(all_seqs)  # 将所有数据打乱顺序
        for each in all_seqs[:num_to_extract]:  # 遍历前80%的数据
            if each == '\n' or each[0] == '>':
                continue
            else:
                seqs.append(each.rstrip())  # 添加非空行且不以'>'开头的数据到列表中

    # data and label
    label = len(seqs) * [lb]
    # TODO TRAIN TEST RATIO
    seqs_train, seqs_test, label_train, label_test = train_test_split(seqs, label, test_size=0.2, random_state=0)
    print('train data:', len(seqs_train))
    print('test data:', len(seqs_test))
    print('train label:', len(label_train))
    print('test_label:', len(label_test))
    print('total numbel:', len(seqs_train)+len(seqs_test))

    return seqs_train, seqs_test, label_train, label_test



def DataClean(data):
    max_len = 0
    for i in range(len(data)):
        st = data[i]
        # get the maximum length of all the sequences
        if(len(st) > max_len): max_len = len(st)

    return data, max_len



def PadEncode(data, max_len):

    # encoding
    amino_acids = 'ACDEFGHIKLMNPRSTUVWY'
    data_e = []
    for i in range(len(data)):
        # print(i)
        length = len(data[i])
        elemt, st = [], data[i]
        for j in st:

            index = amino_acids.index(j)
            elemt.append(index)
        if length < max_len:
            elemt += [0]*(max_len-length)
        data_e.append(elemt)
    # print(".....")
    return data_e



def GetSequenceData(dirs, root):
    # getting training data and test data
    count, max_length = 0, 0
    tr_data, te_data, tr_label, te_label = [], [], [], []
    for dir in dirs:
        # 1.getting data from file
        tr_x, te_x, tr_y, te_y = GetSourceData(root, dir, count)
        count += 1

        # 2.getting the maximum length of all sequences
        tr_x, len_tr = DataClean(tr_x)
        te_x, len_te = DataClean(te_x)
        if len_tr > max_length: max_length = len_tr
        if len_te > max_length: max_length = len_te

        # 3.dataset
        tr_data += tr_x
        te_data += te_x
        tr_label += tr_y
        te_label += te_y


    # data coding and padding vector to the filling length
    traindata = PadEncode(tr_data, max_length)
    testdata = PadEncode(te_data, max_length)

    # data type conversion
    train_data = np.array(traindata)
    test_data = np.array(testdata)
    train_label = np.array(tr_label)
    test_label = np.array(te_label)

    return [train_data, test_data, train_label, test_label]



def GetData(path):
    # TODO DATA PATH
    # dirs = ['AMP', 'ACP', 'ADP', 'AHP', 'AIP'] # functional peptides
    # dirs = ['NEG', 'POS']  # functional peptides
    dirs = ['AntibacterN', 'AntibacterP']
    # dirs = ['AnticancerN', 'AnticancerP']
    # dirs = ['AntimicroN', 'AntimicroP']
    # dirs = ['AVN', 'AVP']
    # dirs = ['TOXN', 'TOXP']
    # dirs = ['ACEN', 'ACEP']
    # dirs = ['AntioxidantN', 'AntioxidantP']
    # dirs = ['AntifungalN', 'AntifungalP']
    # dirs = ['NEUN', 'NEUP']
     # get sequence data
    sequence_data = GetSequenceData(dirs, path)

    return sequence_data



def TrainAndTest(tr_data, tr_label, te_data, te_label):

    from train import train_main # load my training function

    train = [tr_data, tr_label]
    test = [te_data, te_label]

    threshold = 0.5
    model_num = 1 # model number
    test.append(threshold)
    train_main(train, test, model_num, dir)

    ttt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))



def main():
    # I.get sequence data
    path = '.\\Data' # data path
    sequence_data = GetData(path)


    # sequence data partitioning
    tr_seq_data,te_seq_data,tr_seq_label,te_seq_label = \
        sequence_data[0],sequence_data[1],sequence_data[2],sequence_data[3]



    # # II.training and testing
    TrainAndTest(tr_seq_data, tr_seq_label, te_seq_data, te_seq_label)



if __name__ == '__main__':
    # # executing the main function
    for i in range(1):
        main()
    # main()