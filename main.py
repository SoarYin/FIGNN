'''
author: Tengxiao Yin
'''


from loaddata import *
from matplotlib import pyplot as plt
from globals import *
from network import FIGNN


if __name__ == '__main__':
    def training(savename):
        dataset1 = ldata1(0, numsamp)
        dataset2 = ldata0()
        trainrecord = []
        valrecord = []
        for i in range(1, k):
            data1 = ldata1(i, numsamp)
            data2 = ldata2(i)
            dataset1 = dataset1.concatenate(data1)
            dataset2 = dataset2.concatenate(data2)
        data = dataset2.concatenate(dataset1)
        data = data.shuffle(2 * k * numsamp, seed=randomseed)
        valdata = data.skip(1440)
        traindata = data.take(1440)
        traindata = traindata.batch(batch_size=batch_size)
        epo = 300

        for i in range(epo):
            for m in FGCNmodel.metrics:
                m.reset_state()
            print('\nepoch:')
            print(i)
            cont = 1
            for j in traindata:
                print('\r[ %d / %d]' % (cont, traindata.__len__()), end='')
                trainres = FGCNmodel.train_step(j, batch_size=batch_size)
                cont = cont + 1
            print('train_res:')
            tf.print(trainres)
            valres = FGCNmodel.evaluate(valdata)
            trainres_list = []
            for item in trainres.values():
                trainres_list.append(item)
            trainrecord.append(trainres_list)
            valrecord.append(valres)
            if i > 10:
                if i % 5 == 0:
                    FGCNmodel.save_weights('./model/checkpoint' + str(i) + 'model.h5')
            del trainres, valres

        FGCNmodel.save_weights('./record/' + savename + 'model.h5')
        plt.figure(1)
        trainrecord = np.array(trainrecord)
        valrecord = np.array(valrecord)
        np.save('./record/' + savename + 'train.npy', trainrecord)
        np.save('./record/' + savename + 'val.npy', valrecord)
        plt.plot(trainrecord[:, 0], color='darkblue', linestyle='--', label='train_loss')
        plt.plot(valrecord[:, 0], color='darkgreen', linestyle='-', label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig('./record/' + savename + 'fig1.png')
        plt.close()
        plt.figure(2)
        # plt.plot(trainrecord[:, 1], label='accuracy')
        plt.plot(trainrecord[:, 2], color='b', linestyle='--', label='p_train')
        plt.plot(trainrecord[:, 3], color='g', linestyle='--', label='r_train')
        plt.plot(valrecord[:, 2], color='k', linestyle='-', label='p_val')
        plt.plot(valrecord[:, 3], color='m', linestyle='-', label='r_val')
        plt.xlabel('Epoch')
        plt.ylabel('metrics')
        plt.ylim((0, 1))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.legend()
        plt.grid()
        plt.savefig('./record/' + savename + 'fig2.png')
        plt.close()
        plt.figure(3)
        f1_train = 2 * np.multiply(trainrecord[:, 2], trainrecord[:, 3]) / np.add(trainrecord[:, 2],
                                                                                  trainrecord[:, 3]) + 1e-7
        f1_val = 2 * np.multiply(valrecord[:, 2], valrecord[:, 3]) / np.add(valrecord[:, 2], valrecord[:, 3]) + 1e-7
        plt.plot(f1_train, color='b', linestyle='--', label='F1_train')
        plt.plot(f1_val, color='g', linestyle='-', label='F1_val')
        plt.legend()
        plt.ylim((0, 1))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.grid()
        plt.savefig('./record/' + savename + 'fig3.png')
        plt.close()
        #plt.show()
        return
    #准备数据
    k = 21
    modelflag = eval(input('1=training mode, 2=test & confusion, 3=interpreter'))
    numsamp = 50
    if modelflag == 1:
        FGCNmodel = FIGNN(itime=4)
        FGCNmodel.build(input_shape=())
        FGCNmodel.summary()
        learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=1e-0, decay_steps=32, decay_rate=0.1)
        FGCNmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn))
        spr = tf.reduce_mean(FGCNmodel.gl.access).numpy()
        print('persent of paras：' + str(spr)[0:6])
        savename = str(input('enter a save name：'))
        training(savename)

    if modelflag == 2:
        learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=1e-7, decay_steps=8, decay_rate=0.95)
        FGCNmodel = FIGNN(itime=4)
        FGCNmodel.build(input_shape=())
        FGCNmodel.load_weights('./record/GC_itime_4model.h5')
        FGCNmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn))
        dataset1 = ldata1(0, numsamp)
        dataset2 = ldata0()
        for i in range(1, k):
            data1 = ldata1(i, numsamp)
            data2 = ldata2(i)
            dataset1 = dataset1.concatenate(data1)
            dataset2 = dataset2.concatenate(data2)
        data = dataset2.concatenate(dataset1)
        data = data.shuffle(2 * k * numsamp, seed=randomseed)
        valdata = data.skip(1440)
        traindata = data.take(1440)
        FGCNmodel.evaluate(valdata)
        hold = 0.5
        confuse = []
        confuse.append([100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        for i in range(1, k):
            print('class '+str(i)+':')
            FGCNmodel.evaluate(ldata1(i, 50).concatenate(ldata2(i)))
            x = FGCNmodel.predict(ldata1(i, 50).concatenate(ldata2(i)))
            x = np.array(x)
            rex = np.reshape(x, [2*numsamp, k])
            indexes = np.argmax(rex, axis=1)
            pres = np.zeros(shape=[21])
            for i in range(np.size(indexes)):
                if rex[i,indexes[i]] < hold:
                    indexes[i] = 0
            for j in range(np.size(indexes)):
                pres[indexes[j]] = pres[indexes[j]]+1
            confuse.append(pres.tolist())
        confuse = np.array(confuse)
        confuse = np.divide(confuse, 2*numsamp)
        confuse = confuse.T
        plt.figure(3, figsize=(20, 20))
        plt.xlabel('True label')
        plt.ylabel('Predict label')
        plt.xticks(np.arange(k),
                 rotation=45, rotation_mode="anchor", ha="right")
        plt.yticks(np.arange(k),)

        for i in range(k):
            confu = tf.convert_to_tensor(confuse[i, :])
            hold = 0.01
            for j in range(k):
                    if i==j:
                        if round(confuse[i, j], 2) != 0:
                            text = plt.text(j, i, round(confuse[i, j], 2), ha="center", va="center", color="w")
                    elif confuse[i, j]>=hold:
                        if round(confuse[i, j], 2) != 0:
                            text = plt.text(j, i, round(confuse[i, j], 2), ha="center", va="center", color="k")
        plt.imshow(confuse, cmap='Blues')
        plt.colorbar()
        plt.show()


    if modelflag == 3:
        FGCNmodel = FIGNN(itime=4)
        FGCNmodel.build(input_shape=())
        FGCNmodel.load_weights('./record/GC_itime_4model.h5')
        FGCNmodel.compile(optimizer=tf.keras.optimizers.Adam())
        for i in range(1, 21):
            #FGCNmodel.interpreter_stable(name='stab', label=i)
            print(i)
            for k in [5, 15, 25, 35, 45]:
                FGCNmodel.interpreter(ldata2_nl(i, k), name=str(k) + 't', label=i)
                FGCNmodel.interpreter(ldata1_nl(i, k), name=str(k) + 'v', label=i)
