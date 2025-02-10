from Newlabel import *
import tensorflow as tf
import numpy as np
from globals import *

noise = 1e-1


def splitStr(initStr, splitFlag=' '):
    retlist = [ ]
    tmpList = initStr.split(splitFlag)
    #使用spilt(sep = ' ',maxsplit = -1 )分割字符串
    tmpList = list(filter(lambda x: x != '', tmpList))
    #从列表中去除空字符
    for item in tmpList:
        item1 = float(item)
        item2 = [item1]
        retlist.append(item2)
    return retlist

def ldata1(kind,size):
    dataset = []
    refdat = []
    #compute ave
    dataref = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                 '//tennessee-eastman-profBraatz-master//d00_te.dat', 'r')
    for line in dataref:
        refdat.append(splitStr(line))
    dataref.close()
    refdat = np.array(refdat)
    refdat = np.squeeze(refdat)
    refdat = refdat.T

    # 滑动窗口
    avedata = []
    stddata = []
    for j in range(0, 52):
        avedata.append(np.mean(refdat[j]))
        stddata.append((np.std(refdat[j])))
    tempdat = []
    templab = []
    labelset = []
    if kind < 10:
        data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                    '//tennessee-eastman-profBraatz-master//d0'+str(kind)+'_te.dat', 'r')
    else:
        data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                     '//tennessee-eastman-profBraatz-master//d' + str(kind) + '_te.dat', 'r')
    for line in data0:
        tempdat.append(splitStr(line))
    data0.close()
    tempdat = np.array(tempdat)
    tempdat = np.squeeze(tempdat)
    tempdat = tempdat.T
    #滑动窗口
    filtdata0 = []
    deltae0 = []
    for j in range(0, 52):
        tempdata = np.convolve(tempdat[j], np.ones((filtsize,)) / filtsize, mode='valid')
        tempdata = (tempdata-avedata[j])/(stddata[j]) + (2*np.random.rand(np.size(tempdata,0)) - 1)*noise
        filtdata0.append(tempdata)
        delta = np.convolve(tempdata, np.array([1, -1]), mode='valid')
        delta = np.convolve(delta, np.ones(filtsize) / filtsize, mode='valid')
        deltae0.append(delta)
    filtdata0 = np.delete(filtdata0, [0, 1, -1], 1)
    useddata = np.vstack((filtdata0, deltae0))

    templab = Newlabel(templab, kind)
    for re in range(160, len(useddata[0]) - samplen):
        dataset.append(useddata[:, re: re + samplen:2])
        labelset.append(templab)
    dataset = tf.convert_to_tensor(dataset)
    dataset = tf.cast(dataset, dtype=tf.float32)
    #dataset = tf.expand_dims(dataset, axis=-1)
    labelset = tf.convert_to_tensor(labelset)
    data = tf.data.Dataset.from_tensor_slices((dataset, labelset))
    #data = data.shuffle(buffer_size=len(labelset))
    data = data.take(size)
    return data

def ldata2(kind):
    np.random.seed(10)
    dataset = []
    refdat = []
    # compute ave
    dataref = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                   '//tennessee-eastman-profBraatz-master//d00.dat', 'r')
    for line in dataref:
        refdat.append(splitStr(line))
    dataref.close()
    refdat = np.array(refdat)
    refdat = np.squeeze(refdat)
    #refdat = refdat.T

    # 滑动窗口
    avedata = []
    stddata = []
    for j in range(0, 52):
        avedata.append(np.mean(refdat[j]))
        stddata.append((np.std(refdat[j])))
    tempdat = []
    templab = []
    labelset = []
    if kind < 10:
        data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                     '//tennessee-eastman-profBraatz-master//d0' + str(kind) + '.dat', 'r')
    else:
        data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                     '//tennessee-eastman-profBraatz-master//d' + str(kind) + '.dat', 'r')
    for line in data0:
        tempdat.append(splitStr(line))
    data0.close()
    tempdat = np.array(tempdat)
    tempdat = np.squeeze(tempdat)
    tempdat = tempdat.T.tolist()
    # 滑动窗口
    filtdata0 = []
    deltae0 = []
    for j in range(0, 52):
        tempdata = np.convolve(tempdat[j], np.ones((filtsize,)) / filtsize, mode='valid')
        tempdata = (tempdata - avedata[j]) / (stddata[j]) + (2*np.random.rand(np.size(tempdata,0)) - 1)*noise
        filtdata0.append(tempdata)
        delta = np.convolve(tempdata, np.array([1, -1]), mode='valid')
        delta = np.convolve(delta, np.ones(filtsize) / filtsize, mode='valid')
        deltae0.append(delta)
    filtdata0 = np.delete(filtdata0, [0, 1, -1], 1)
    useddata = np.vstack((filtdata0, deltae0))

    templab = Newlabelt(templab, kind)
    templab = np.array(templab)
    templab = templab.T
    for re in range(160, len(useddata[0]) - samplen):
        dataset.append(useddata[:, re: re + samplen:2])
        labelset.append(templab)
    dataset = tf.convert_to_tensor(dataset)
    dataset = tf.cast(dataset, dtype=tf.float32)
    #dataset = tf.expand_dims(dataset, axis=-1)
    labelset = tf.convert_to_tensor(labelset)
    data = tf.data.Dataset.from_tensor_slices((dataset, labelset))
    # data = data.shuffle(buffer_size=len(labelset))
    data = data.take(50)
    return data

def ldata0():
    kind = 0
    dataset = []
    refdat = []
    # compute ave
    dataref = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                   '//tennessee-eastman-profBraatz-master//d00.dat', 'r')
    for line in dataref:
        refdat.append(splitStr(line))
    dataref.close()
    refdat = np.array(refdat)
    refdat = np.squeeze(refdat)
    #refdat = refdat.T

    # 滑动窗口
    avedata = []
    stddata = []
    for j in range(0, 52):
        avedata.append(np.mean(refdat[j]))
        stddata.append((np.std(refdat[j])))
    tempdat = []
    templab = []
    labelset = []
    data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                     '//tennessee-eastman-profBraatz-master//d00.dat', 'r')
    for line in data0:
        tempdat.append(splitStr(line))
    data0.close()
    artempdat = np.array(tempdat)

    tempdat = np.squeeze(artempdat)
    # 滑动窗口
    filtdata0 = []
    deltae0 = []
    for j in range(0, 52):
        tempdata = np.convolve(tempdat[j], np.ones((filtsize,)) / filtsize, mode='valid')
        tempdata = (tempdata - avedata[j]) / (stddata[j])
        filtdata0.append(tempdata)
        delta = np.convolve(tempdata, np.array([1, -1]), mode='valid')
        delta = np.convolve(delta, np.ones(filtsize) / filtsize, mode='valid')
        deltae0.append(delta)
    filtdata0 = np.delete(filtdata0, [0, 1, -1], 1)
    useddata = np.vstack((filtdata0, deltae0))

    templab = Newlabelt(templab, kind)
    templab = np.array(templab)
    templab = templab.T
    for re in range(160, len(useddata[0]) - samplen):
        dataset.append(useddata[:, re: re + samplen:2])
        labelset.append(templab)
    dataset = tf.convert_to_tensor(dataset, dtype=tf.float32)
    #dataset = tf.expand_dims(dataset, axis=-1)
    labelset = tf.convert_to_tensor(labelset)
    data = tf.data.Dataset.from_tensor_slices((dataset, labelset))
    # data = data.shuffle(buffer_size=len(labelset))
    data = data.take(50)
    return data


def ldata1_nl(kind,loc):
    dataset = []
    refdat = []
    #compute ave
    dataref = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                 '//tennessee-eastman-profBraatz-master//d00_te.dat', 'r')
    for line in dataref:
        refdat.append(splitStr(line))
    dataref.close()
    refdat = np.array(refdat)
    refdat = np.squeeze(refdat)
    refdat = refdat.T

    # 滑动窗口
    avedata = []
    stddata = []
    for j in range(0, 52):
        avedata.append(np.mean(refdat[j]))
        stddata.append((np.std(refdat[j])))
    tempdat = []
    templab = []
    labelset = []
    if kind < 10:
        data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                    '//tennessee-eastman-profBraatz-master//d0'+str(kind)+'_te.dat', 'r')
    else:
        data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                     '//tennessee-eastman-profBraatz-master//d' + str(kind) + '_te.dat', 'r')
    for line in data0:
        tempdat.append(splitStr(line))
    data0.close()
    tempdat = np.array(tempdat)
    tempdat = np.squeeze(tempdat)
    tempdat = tempdat.T
    #滑动窗口
    filtdata0 = []
    deltae0 = []
    for j in range(0, 52):
        tempdata = np.convolve(tempdat[j], np.ones((filtsize,)) / filtsize, mode='valid')
        tempdata = (tempdata-avedata[j])/(stddata[j])
        filtdata0.append(tempdata)
        delta = np.convolve(tempdata, np.array([1, -1]), mode='valid')
        delta = np.convolve(delta, np.ones(filtsize) / filtsize, mode='valid')
        deltae0.append(delta)
    filtdata0 = np.delete(filtdata0, [0, 1, -1], 1)
    useddata = np.vstack((filtdata0, deltae0))

    templab = Newlabel(templab, kind)
    for re in range(160, len(useddata[0]) - samplen):
        dataset.append(useddata[:, re: re + samplen:2])
        labelset.append(templab)
    dataset = tf.convert_to_tensor(dataset)
    dataset = tf.cast(dataset, dtype=tf.float32)
    return dataset[loc]



def ldata2_nl(kind, loc):
    dataset = []
    refdat = []
    # compute ave
    dataref = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                   '//tennessee-eastman-profBraatz-master//d00.dat', 'r')
    for line in dataref:
        refdat.append(splitStr(line))
    dataref.close()
    refdat = np.array(refdat)
    refdat = np.squeeze(refdat)
    #refdat = refdat.T

    # 滑动窗口
    avedata = []
    stddata = []
    for j in range(0, 52):
        avedata.append(np.mean(refdat[j]))
        stddata.append((np.std(refdat[j])))
    tempdat = []
    templab = []
    labelset = []
    if kind < 10:
        data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                     '//tennessee-eastman-profBraatz-master//d0' + str(kind) + '.dat', 'r')
    else:
        data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                     '//tennessee-eastman-profBraatz-master//d' + str(kind) + '.dat', 'r')
    for line in data0:
        tempdat.append(splitStr(line))
    data0.close()
    tempdat = np.array(tempdat)
    tempdat = np.squeeze(tempdat)
    if kind!=0:
        tempdat = tempdat.T.tolist()
    # 滑动窗口
    filtdata0 = []
    deltae0 = []
    for j in range(0, 52):
        tempdata = np.convolve(tempdat[j], np.ones((filtsize,)) / filtsize, mode='valid')
        tempdata = (tempdata - avedata[j]) / (stddata[j])
        filtdata0.append(tempdata)
        delta = np.convolve(tempdata, np.array([1, -1]), mode='valid')
        delta = np.convolve(delta, np.ones(filtsize) / filtsize, mode='valid')
        deltae0.append(delta)
    filtdata0 = np.delete(filtdata0, [0, 1, -1], 1)
    useddata = np.vstack((filtdata0, deltae0))

    templab = Newlabelt(templab, kind)
    templab = np.array(templab)
    templab = templab.T
    for re in range(160, len(useddata[0]) - samplen):
        dataset.append(useddata[:, re: re + samplen:2])
        labelset.append(templab)
    dataset = tf.convert_to_tensor(dataset)
    dataset = tf.cast(dataset, dtype=tf.float32)
    return dataset[loc]


