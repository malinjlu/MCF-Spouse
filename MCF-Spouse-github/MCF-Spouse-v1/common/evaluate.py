# -*- coding: utf-8 -*-


# this code has been checked and all function are no problem

# author:zesen.chen

# email:seu_czs@126.com

# Checked on Mon Feb 27 2017


import numpy as np
import scipy.io as sci

def find(instance, label1, label2):
    index1 = []
    index2 = []

    for i in range(instance.shape[0]):
        if instance[i] == label1:
            index1.append(i)
        if instance[i] == label2:
            index2.append(i)
    return index1, index2


def findmax(outputs):
    Max = -float("inf")
    index = 0
    for i in range(outputs.shape[0]):
        if outputs[i] > Max:
            Max = outputs[i]
            index = i
    return Max, index


def sort(x):
    temp = np.array(x)
    length = temp.shape[0]
    index = []
    sortX = []
    for i in range(length):
        Min = float("inf")
        Min_j = i
        for j in range(length):
            if temp[j] < Min:
                Min = temp[j]
                Min_j = j
        sortX.append(Min)
        index.append(Min_j)
        temp[Min_j] = float("inf")
    return temp, index


def findIndex(a, b):
    for i in range(len(b)):
        if a == b[i]:
            return i


def avgprec(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []

    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(outputs[i])
            temp_test_target.append(test_target[i])
            labels_size.append(sum(test_target[i] == 1))
            index1, index2 = find(test_target[i], 1, 0)
            labels_index.append(index1)
            not_labels_index.append(index2)

    aveprec = 0

    for i in range(instance_num):
        tempvalue, index = sort(temp_outputs[i])
        indicator = np.zeros((class_num,))
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            indicator[loc] = 1
        summary = 0

        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            # print(loc)
            summary = summary + sum(indicator[loc:class_num]) / (class_num - loc);
        aveprec = aveprec + summary / labels_size[i]

    return aveprec / test_data_num


def Coverage(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    labels_index = []
    not_labels_index = []
    labels_size = []

    for i in range(test_data_num):
        labels_size.append(sum(test_target[i] == 1))
        index1, index2 = find(test_target[i], 1, 0)
        labels_index.append(index1)
        not_labels_index.append(index2)

    cover = 0

    for i in range(test_data_num):
        tempvalue, index = sort(outputs[i])
        temp_min = class_num + 1

        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            if loc < temp_min:
                temp_min = loc

        cover = cover + (class_num - temp_min)
    # return (cover / test_data_num - 1)
    return (cover / test_data_num - 1) / class_num


def HammingLoss(predict_labels, test_target):
    labels_num = predict_labels.shape[1]
    test_data_num = predict_labels.shape[0]
    hammingLoss = 0

    for i in range(test_data_num):
        notEqualNum = 0

        for j in range(labels_num):
            if predict_labels[i][j] != test_target[i][j]:
                notEqualNum = notEqualNum + 1

        hammingLoss = hammingLoss + notEqualNum / labels_num

    hammingLoss = hammingLoss / test_data_num

    return hammingLoss

def rloss(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []

    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(outputs[i])
            temp_test_target.append(test_target[i])
            labels_size.append(sum(test_target[i] == 1))
            index1, index2 = find(test_target[i], 1, 0)
            labels_index.append(index1)
            not_labels_index.append(index2)

    rankloss = 0

    for i in range(instance_num):
        m = labels_size[i]
        n = class_num - m
        temp = 0
        for j in range(m):
            for k in range(n):
                if temp_outputs[i][labels_index[i][j]] < temp_outputs[i][not_labels_index[i][k]]:
                    temp = temp + 1

        rankloss = rankloss + temp / (m * n)

    rankloss = rankloss / instance_num

    return rankloss


def SubsetAccuracy(predict_labels, test_target):
    test_data_num = predict_labels.shape[0]
    class_num = predict_labels.shape[1]
    correct_num = 0

    for i in range(test_data_num):
        for j in range(class_num):
            if predict_labels[i][j] != test_target[i][j]:
                break

        if j == class_num - 1:
            correct_num = correct_num + 1

    return correct_num / test_data_num



def Performance(predict_labels, test_target):
    data_num = predict_labels.shape[0]
    tempPre = np.transpose(np.copy(predict_labels))
    tempTar = np.transpose(np.copy(test_target))
    tempTar[tempTar == 0] = -1
    com = sum(tempPre == tempTar)
    tempTar[tempTar == -1] = 0
    PreLab = sum(tempPre)
    TarLab = sum(tempTar)

    I = 0

    for i in range(data_num):
        if TarLab[i] == 0:
            I += 1
        else:
            if PreLab[i] == 0:
                I += 0
            else:
                I += com[i] / PreLab[i]

    return I / data_num


def DatasetInfo(filename):
    Dict = sci.loadmat(filename)
    data = Dict['data']
    target = Dict['target']
    data_num = data.shape[0]
    dim = data.shape[1]
    if target.shape[0] != data_num:
        target = np.transpose(target)
    labellen = target.shape[1]
    attr = 'numeric'

    if np.max(data) == 1 and np.min(data) == 0:
        attr = 'nominal'

    if np.min(target) == -1:
        target[target == -1] = 0

    target = np.transpose(target)
    LCard = sum(sum(target)) / data_num
    LDen = LCard / labellen
    labellist = []

    for i in range(data_num):
        if list(target[:, i]) not in labellist:
            labellist.append(list(target[:, i]))

    LDiv = len(labellist)
    PLDiv = LDiv / data_num
    print('|S|:', data_num)
    print('dim(S):', dim)
    print('L(S):', labellen)
    print('F(S):', attr)
    print('LCard(S):', LCard)
    print('LDen(S):', LDen)
    print('LDiv(S):', LDiv)
    print('PLDiv(S):', PLDiv)


def Friedman(N, k, r):
    r2 = [r[i] ** 2 for i in range(k)]
    temp = (sum(r2) - k * ((k + 1) ** 2) / 4) * 12 * N / k / (k + 1)
    F = (N - 1) * temp / (N * (k - 1) - temp)

    return F


def Fmacro(predict_labels, test_target):
    label_number = predict_labels.shape[1]
    test_number = test_target.shape[0]

    TP_ = [0 for i in range(label_number)]
    FP_ = [0 for i in range(label_number)]
    FN_ = [0 for i in range(label_number)]
    for j in range(label_number):

        for i in range(test_number):
            if predict_labels[i][j] == 1 and test_target[i][j] == 1:
                TP_[j] = TP_[j] + 1
            if predict_labels[i][j] == 1 and test_target[i][j] == 0:
                FP_[j] = FP_[j] + 1
            if predict_labels[i][j] == 0 and test_target[i][j] == 1:
                FN_[j] = FN_[j] + 1

    Fma = 0
    for i in range(label_number):
        a = 2*TP_[i]*1.0
        b = (FP_[i] + a + FN_[i])
        if b != 0:
            c = a/b
            Fma = Fma + c
    Fma = Fma/label_number
    return Fma


def Fmicro(predict_labels, test_target):
    label_number = predict_labels.shape[1]
    test_number = test_target.shape[0]

    TP_ = [0 for i in range(label_number)]
    FP_ = [0 for i in range(label_number)]
    FN_ = [0 for i in range(label_number)]
    for j in range(label_number):
        for i in range(test_number):
            if predict_labels[i][j] == 1 and test_target[i][j] == 1:
                TP_[j] = TP_[j] + 1
            if predict_labels[i][j] == 1 and test_target[i][j] == 0:
                FP_[j] = FP_[j] + 1
            if predict_labels[i][j] == 0 and test_target[i][j] == 1:
                FN_[j] = FN_[j] + 1

    TP = 0
    FP = 0
    FN = 0

    for i in range(label_number):
        TP = TP + TP_[i]
        FP = FP + FP_[i]
        FN = FN + FN_[i]
    TP = TP*2*1.0
    sum = (TP + FP + FN)*1.0
    Fmi = TP/sum
    return Fmi