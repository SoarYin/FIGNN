import numpy as np
from loaddata import *
import pandas as pd
from normalize import normalize
from statsmodels.tsa.stattools import grangercausalitytests
def k_ca():
    df = pd.read_excel('prior_knowledge.xlsx', header=None)
    array = df.to_numpy()
    array = np.hstack((array,array))
    array = np.vstack((array,array))
    matrix_without_1 = np.where(array != 1, array, 0)
    matrix_without_1 = np.abs(matrix_without_1)
    matrix_without_minus_1 = np.where(array != -1, array, 0)
    df2 = pd.read_excel('knowledge2.xlsx', header=None)
    array2 = df2.to_numpy()
    pos_matrix = np.where(np.logical_or(array2 == 1, array2 == 2), 1, 0)
    neg_matrix = np.where(np.logical_or(array2 == -1, array2 == 2), 1, 0)
    hstack_mat = np.hstack((pos_matrix, neg_matrix))
    print(np.linalg.matrix_rank(hstack_mat))
    fix1 = np.ones([1, 208])
    fix2 = np.ones([19, 208])
    stack_mat = np.vstack((fix1, hstack_mat, fix2))
    n1 = np.eye(21, dtype=np.float32)
    n2 = np.zeros([21, 19], dtype=np.float32)
    n3 = np.zeros([19, 21], dtype=np.float32)
    n4 = np.ones([19, 19], dtype=np.float32)
    n5 = np.hstack([n1, n2])
    n6 = np.hstack([n3, n4])
    n7 = np.vstack([n5, n6])
    corr_lab = tf.concat([stack_mat, tf.convert_to_tensor(n7, dtype=tf.float32)], axis=1)
    corr_var1 = tf.concat([matrix_without_minus_1, matrix_without_1, tf.zeros(shape=[104, 40])], axis=1)
    corr_var2 = tf.concat([matrix_without_1, matrix_without_minus_1, tf.zeros(shape=[104, 40])], axis=1)
    corr_var = tf.concat([corr_var1, corr_var2], axis=0)
    corr = tf.concat([tf.cast(corr_var, dtype=tf.float32), corr_lab], axis=0)
    corr_out = tf.convert_to_tensor(corr)
    return corr_out
def kca_w():
    w = k_ca()
    noise = tf.random.uniform(minval=-4e-1, maxval=4e-1, shape=[248, 248], seed=23, dtype=tf.float32,)
    wn = w + noise
    return wn
def adj_ca():
    arrays = []
    for k in range(0, 21):
        refdat = []
        # compute ave
        dataref = open('./data/tennessee-eastman-profBraatz-master/d00_te.dat', 'r')
        for line in dataref:
            refdat.append(splitStr(line))
        dataref.close()
        refdat = np.array(refdat)
        refdat = np.squeeze(refdat)
        refdat = refdat.T
        avedata = []
        stddata = []
        for j in range(0, 52):
            avedata.append(np.mean(refdat[j]))
            stddata.append((np.std(refdat[j])))
        tempdat = []
        if k < 10:
            data0 = open('./data/tennessee-eastman-profBraatz-master//d0' + str(k) + '_te.dat', 'r')
        else:
            data0 = open('./data/tennessee-eastman-profBraatz-master//d' + str(k) + '_te.dat', 'r')
        for line in data0:
            tempdat.append(splitStr(line))
        data0.close()
        tempdat = np.array(tempdat)
        tempdat = np.squeeze(tempdat)
        tempdat = tempdat.T
        filtdata0 = []
        deltae0 = []
        for j in range(0, 52):
            tempdata = np.convolve(tempdat[j], np.ones((filtsize,)) / filtsize, mode='valid')
            tempdata = (tempdata - avedata[j]) / (stddata[j])
            filtdata0.append(tempdata)
            ave = np.mean(tempdata)
            delta = np.convolve(tempdata, np.array([1, -1]), mode='valid')
            delta = np.convolve(delta, np.ones(filtsize) / filtsize, mode='valid')
            deltae0.append(delta)
        filtdata0 = np.delete(filtdata0, [0, 1, -1], 1)
        useddata = np.vstack((filtdata0, deltae0))
        arrays.append(useddata)
    corr_lab = np.ones([40, 104])
    corr_var = np.eye(104)
    arrayrestore = np.array(arrays)
    arrayactive = arrayrestore[:, :, 160:310]
    for i in range(1, 21):
        for j in range(0, 104):
            cos_sim = arrayactive[i, j, :].flatten().dot(arrayactive[0,j,:].flatten()) \
                      / (np.linalg.norm(arrayactive[i, j, :].flatten()) * np.linalg.norm(arrayactive[0,j,:].flatten()))
            corr_lab[i, j] = 1 - abs(cos_sim)
    n1 = np.eye(21, dtype=np.float32)
    n2 = np.zeros([21, 19], dtype=np.float32)
    n3 = np.zeros([19, 21], dtype=np.float32)
    n4 = np.ones([19, 19], dtype=np.float32)
    n5 = np.hstack([n1, n2])
    n6 = np.hstack([n3, n4])
    n7 = np.vstack([n5, n6])
    corr_lab = tf.concat([corr_lab, corr_lab, tf.convert_to_tensor(n7, dtype=tf.float32)], axis=1)
    for m in range(0, 104):
        for n in range(0, 104):
            cos_sim = arrayactive[1:, m, :].flatten().dot(arrayactive[1:, n, :].flatten()) \
                      / (np.linalg.norm(arrayactive[1:, m, :].flatten()) * np.linalg.norm(
                arrayactive[1:, n, :].flatten()))
            corr_var[m, n] = cos_sim
    posmask = tf.where(corr_var <= 0, x=0.0, y=1.0)
    negmask = tf.where(corr_var <= 0, x=-1.0, y=0.0)
    pos = tf.multiply(posmask, corr_var)
    neg = tf.multiply(negmask, corr_var)
    corr_var1 = tf.concat([pos, neg, tf.zeros(shape=[104, 40])], axis=1)
    corr_var2 = tf.concat([neg, pos, tf.zeros(shape=[104, 40])], axis=1)
    corr_var = tf.concat([corr_var1, corr_var2], axis=0)
    corr = tf.concat([tf.cast(corr_var, dtype=tf.float32), corr_lab], axis=0)
    corr_out = tf.convert_to_tensor(corr)
    corr_out = tf.where(corr_out > 0.5, x=1.0, y=0.0)
    return corr_out

def adj_w():

    arrays = []
    for k in range(0, 21):
        refdat = []
        # compute ave
        dataref = open('./data/tennessee-eastman-profBraatz-master//d00_te.dat', 'r')
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
        if k < 10:
            data0 = open('./data/tennessee-eastman-profBraatz-master//d0' + str(k) + '_te.dat', 'r')
        else:
            data0 = open('./data/tennessee-eastman-profBraatz-master//d' + str(k) + '_te.dat', 'r')
        for line in data0:
            tempdat.append(splitStr(line))
        data0.close()
        tempdat = np.array(tempdat)
        tempdat = np.squeeze(tempdat)
        tempdat = tempdat.T
        # 滑动窗口
        filtdata0 = []
        deltae0 = []
        for j in range(0, 52):
            tempdata = np.convolve(tempdat[j], np.ones((filtsize,)) / filtsize, mode='valid')
            tempdata = (tempdata - avedata[j]) / (stddata[j])
            filtdata0.append(tempdata)
            ave = np.mean(tempdata)
            delta = np.convolve(tempdata, np.array([1, -1]), mode='valid')
            delta = np.convolve(delta, np.ones(filtsize) / filtsize, mode='valid')
            deltae0.append(delta)
        filtdata0 = np.delete(filtdata0, [0, 1, -1], 1)
        useddata = np.vstack((filtdata0, deltae0))
        arrays.append(useddata)
    corr_lab = np.ones([40, 104])
    corr_var = np.eye(104)
    arrayrestore = np.array(arrays)
    arrayactive = arrayrestore[:, :, 160:310]
    # 依据是：如果一个变量和正常变量相似，那么该变量对于最终的错误报警应当没有贡献
    for i in range(1, 21):
        for j in range(0, 104):
            cos_sim = arrayactive[i, j, :].flatten().dot(arrayactive[0, j, :].flatten()) \
                      / (np.linalg.norm(arrayactive[i, j, :].flatten()) * np.linalg.norm(
                arrayactive[0, j, :].flatten()))
            corr_lab[i, j] = 1 - abs(cos_sim)
    n1 = np.eye(21, dtype=np.float32)
    n2 = np.zeros([21, 19], dtype=np.float32)
    n3 = np.zeros([19, 21], dtype=np.float32)
    n4 = np.ones([19, 19], dtype=np.float32)
    n5 = np.hstack([n1, n2])
    n6 = np.hstack([n3, n4])
    n7 = np.vstack([n5, n6])
    corr_lab = tf.concat([corr_lab, corr_lab, tf.convert_to_tensor(n7, dtype=tf.float32)], axis=1)
    for m in range(0, 104):
        for n in range(0, 104):
            cos_sim = arrayactive[1:, m, :].flatten().dot(arrayactive[1:, n, :].flatten()) \
                      / (np.linalg.norm(arrayactive[1:, m, :].flatten()) * np.linalg.norm(
                arrayactive[1:, n, :].flatten()))
            corr_var[m, n] = abs(cos_sim)
    corr_var = tf.concat([corr_var, corr_var, tf.zeros(shape=[104, 40])], axis=1)
    corr_var = tf.concat([corr_var, corr_var], axis=0)
    corr = tf.concat([tf.cast(corr_var, dtype=tf.float32), corr_lab], axis=0)
    corr_out = tf.convert_to_tensor(corr)
    return corr_out
def adj_causal():
    arrays = []
    for k in range(0, 21):
        refdat = []
        dataref = open('./data/tennessee-eastman-profBraatz-master//d00_te.dat', 'r')
        for line in dataref:
            refdat.append(splitStr(line))
        dataref.close()
        refdat = np.array(refdat)
        refdat = np.squeeze(refdat)
        refdat = refdat.T
        avedata = []
        stddata = []
        for j in range(0, 52):
            avedata.append(np.mean(refdat[j]))
            stddata.append((np.std(refdat[j])))
        tempdat = []
        if k < 10:
            data0 = open('./data/tennessee-eastman-profBraatz-master//d0' + str(k) + '_te.dat', 'r')
        else:
            data0 = open('./data/tennessee-eastman-profBraatz-master//d' + str(k) + '_te.dat', 'r')
        for line in data0:
            tempdat.append(splitStr(line))
        data0.close()
        tempdat = np.array(tempdat)
        tempdat = np.squeeze(tempdat)
        tempdat = tempdat.T
        filtdata0 = []
        deltae0 = []
        for j in range(0, 52):
            tempdata = np.convolve(tempdat[j], np.ones((filtsize,)) / filtsize, mode='valid')
            tempdata = (tempdata - avedata[j]) / (stddata[j])
            filtdata0.append(tempdata)
            ave = np.mean(tempdata)
            delta = np.convolve(tempdata, np.array([1, -1]), mode='valid')
            delta = np.convolve(delta, np.ones(filtsize) / filtsize, mode='valid')
            deltae0.append(delta)
        filtdata0 = np.delete(filtdata0, [0, 1, -1], 1)
        useddata = np.vstack((filtdata0, deltae0))
        arrays.append(useddata)
    corr_lab = np.ones([40, 104])
    corr_var = np.eye(104)
    arrayrestore = np.array(arrays)
    arrayactive = arrayrestore[:, :, 160:310]
    for i in range(1, 21):
        for j in range(0, 104):
            maxlag = 1  # 滞后阶数
            test_results = grangercausalitytests(np.column_stack((arrayactive[i, j, :].flatten(),
                                                                  arrayactive[0, j, :].flatten())), maxlag)
            corr_lab[i, j] = abs(test_results[1][0]["ssr_ftest"][1])
    n1 = np.eye(21, dtype=np.float32)
    n2 = np.zeros([21, 19], dtype=np.float32)
    n3 = np.zeros([19, 21], dtype=np.float32)
    n4 = np.ones([19, 19], dtype=np.float32)
    n5 = np.hstack([n1, n2])
    n6 = np.hstack([n3, n4])
    n7 = np.vstack([n5, n6])
    corr_lab = tf.concat([corr_lab, corr_lab, tf.convert_to_tensor(n7, dtype=tf.float32)], axis=1)
    corr_lab = tf.where(corr_lab > 0.1, x=1.0, y=0.0)
    for m in range(0, 104):
        for n in range(0, 104):
            cos_sim = arrayactive[1:, m, :].flatten().dot(arrayactive[1:, n, :].flatten()) \
                      / (np.linalg.norm(arrayactive[1:, m, :].flatten()) * np.linalg.norm(
                arrayactive[1:, n, :].flatten()))
            maxlag = 1  # 滞后阶数
            test_results = grangercausalitytests(np.column_stack((arrayactive[1:, m, :].flatten(),
                                                                  arrayactive[1:, n, :].flatten())), maxlag)
            corr_var[m, n] =abs(test_results[1][0]["ssr_ftest"][1])
    corr_var = tf.where(corr_var < 1e-7, x=1.0, y=0.0)
    corr_var1 = tf.concat([corr_var, corr_var, tf.zeros(shape=[104, 40])], axis=1)
    corr_var2 = tf.concat([corr_var, corr_var, tf.zeros(shape=[104, 40])], axis=1)
    corr_var = tf.concat([corr_var1, corr_var2], axis=0)
    corr = tf.concat([tf.cast(corr_var, dtype=tf.float32), corr_lab], axis=0)
    corr_out = tf.convert_to_tensor(corr)
    return corr_out
def adj_causal_w():
    arrays = []
    for k in range(0, 21):
        refdat = []
        # compute ave
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
        if k < 10:
            data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                         '//tennessee-eastman-profBraatz-master//d0' + str(k) + '_te.dat', 'r')
        else:
            data0 = open('D://程序项目//Rule And Data Fusion Deeplearning For Industrial fault classfication'
                         '//tennessee-eastman-profBraatz-master//d' + str(k) + '_te.dat', 'r')
        for line in data0:
            tempdat.append(splitStr(line))
        data0.close()
        tempdat = np.array(tempdat)
        tempdat = np.squeeze(tempdat)
        tempdat = tempdat.T
        # 滑动窗口
        filtdata0 = []
        deltae0 = []
        for j in range(0, 52):
            tempdata = np.convolve(tempdat[j], np.ones((filtsize,)) / filtsize, mode='valid')
            tempdata = (tempdata - avedata[j]) / (stddata[j])
            filtdata0.append(tempdata)
            ave = np.mean(tempdata)
            delta = np.convolve(tempdata, np.array([1, -1]), mode='valid')
            delta = np.convolve(delta, np.ones(filtsize) / filtsize, mode='valid')
            deltae0.append(delta)
        filtdata0 = np.delete(filtdata0, [0, 1, -1], 1)
        useddata = np.vstack((filtdata0, deltae0))
        arrays.append(useddata)
    corr_lab = np.ones([40, 104])
    corr_var = np.eye(104)
    arrayrestore = np.array(arrays)
    arrayactive = arrayrestore[:, :, 160:310]
    # 依据是：如果一个变量和正常变量相似，那么该变量对于最终的错误报警应当没有贡献
    for i in range(1, 21):
        for j in range(0, 104):
            maxlag = 1  # 滞后阶数
            test_results = grangercausalitytests(np.column_stack((arrayactive[i, j, :].flatten(),
                                                                  arrayactive[0, j, :].flatten())), maxlag)
            corr_lab[i, j] = abs(test_results[1][0]["ssr_ftest"][0])
    n1 = np.eye(21, dtype=np.float32)
    n2 = np.zeros([21, 19], dtype=np.float32)
    n3 = np.zeros([19, 21], dtype=np.float32)
    n4 = np.ones([19, 19], dtype=np.float32)
    n5 = np.hstack([n1, n2])
    n6 = np.hstack([n3, n4])
    n7 = np.vstack([n5, n6])
    corr_lab = tf.concat([corr_lab, corr_lab, tf.convert_to_tensor(n7, dtype=tf.float32)], axis=1)
    for m in range(0, 104):
        for n in range(0, 104):
            cos_sim = arrayactive[1:, m, :].flatten().dot(arrayactive[1:, n, :].flatten()) \
                      / (np.linalg.norm(arrayactive[1:, m, :].flatten()) * np.linalg.norm(
                arrayactive[1:, n, :].flatten()))
            maxlag = 1  # 滞后阶数
            test_results = grangercausalitytests(np.column_stack((arrayactive[1:, m, :].flatten(),
                                                                  arrayactive[1:, n, :].flatten())), maxlag)
            corr_var[m, n] =1 - abs(test_results[1][0]["ssr_ftest"][0])
    # 2.建立正数的mask和负数的mask
    posmask = tf.where(corr_var <= 0, x=0.0, y=1.0)
    negmask = tf.where(corr_var <= 0, x=-1.0, y=0.0)
    # 3.mask掉不需要的阈值
    pos = tf.multiply(posmask, corr_var)
    neg = tf.multiply(negmask, corr_var)
    corr_var1 = tf.concat([pos, neg, tf.zeros(shape=[104, 40])], axis=1)
    corr_var2 = tf.concat([neg, pos, tf.zeros(shape=[104, 40])], axis=1)
    corr_var = tf.concat([corr_var1, corr_var2], axis=0)
    corr = tf.concat([tf.cast(corr_var, dtype=tf.float32), corr_lab], axis=0)
    corr_out = tf.convert_to_tensor(corr)
    return corr_out

def full_conection():
    corr = np.ones([248, 248])
    corr_out = tf.convert_to_tensor(corr)
    corr_out = tf.cast(corr_out, dtype=tf.float32)
    return corr_out

def random_weight():
    np.random.seed(23)
    corr = np.random.uniform(0,1,size=[248, 248])
    corr_out = tf.convert_to_tensor(corr)
    corr_out = tf.cast(corr_out, dtype=tf.float32)
    return corr_out

def adj_PTCN():
    arrays = []
    for k in range(0, 21):
        refdat = []

        dataref = open('./data/tennessee-eastman-profBraatz-master//d00_te.dat', 'r')
        for line in dataref:
            refdat.append(splitStr(line))
        dataref.close()
        refdat = np.array(refdat)
        refdat = np.squeeze(refdat)
        refdat = refdat.T
        avedata = []
        stddata = []
        for j in range(0, 52):
            avedata.append(np.mean(refdat[j]))
            stddata.append((np.std(refdat[j])))
        tempdat = []
        if k < 10:
            data0 = open('./data/tennessee-eastman-profBraatz-master//d0' + str(k) + '_te.dat', 'r')
        else:
            data0 = open('./data/tennessee-eastman-profBraatz-master//d' + str(k) + '_te.dat', 'r')
        for line in data0:
            tempdat.append(splitStr(line))
        data0.close()
        tempdat = np.array(tempdat)
        tempdat = np.squeeze(tempdat)
        tempdat = tempdat.T

        filtdata0 = []
        deltae0 = []
        for j in range(0, 52):
            tempdata = np.convolve(tempdat[j], np.ones((filtsize,)) / filtsize, mode='valid')
            tempdata = (tempdata - avedata[j]) / (stddata[j])
            filtdata0.append(tempdata)
            ave = np.mean(tempdata)
            delta = np.convolve(tempdata, np.array([1, -1]), mode='valid')
            delta = np.convolve(delta, np.ones(filtsize) / filtsize, mode='valid')
            deltae0.append(delta)
        filtdata0 = np.delete(filtdata0, [0, 1, -1], 1)
        useddata = np.vstack((filtdata0, deltae0))
        arrays.append(useddata)
    corr_lab = np.ones([40, 104])
    corr_var = np.eye(104)
    arrayrestore = np.array(arrays)
    arrayactive = arrayrestore[:, :, 160:310]

    for m in range(0, 104):
        for n in range(0, 104):
            cos_sim = arrayactive[1:, m, :].flatten().dot(arrayactive[1:, n, :].flatten()) \
                      / (np.linalg.norm(arrayactive[1:, m, :].flatten()) * np.linalg.norm(
                arrayactive[1:, n, :].flatten()))
            corr_var[m, n] = abs(cos_sim)
    A_bar = tf.where(corr_var >= 0.3, x=1.0, y=0.0)
    D = tf.sqrt(tf.linalg.diag(tf.reduce_sum(A_bar, axis=1)),)
    invd = tf.linalg.inv(D)
    res = tf.matmul(invd, A_bar)
    res = tf.matmul(res, invd)
    return res




if __name__ == '__main__':
    g = adj_PTCN()


