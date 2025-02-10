
import tensorflow as tf
import numpy as np
from normalize import normalize, invnorm
from globals import *
import gc
import graphconstruct
import graphviz as gz

def O2P(a):
    x = -1
    mask1 = tf.where(a <= 0.5, x=1.0, y=0.0)
    mask2 = tf.where(a > 0.5, x=1.0, y=0.0)
    ap1 = (- (26 * x) / 7 - 13 / 5) * tf.pow(a, 3) + ((13 * x) / 7 + 27 / 10) * tf.pow(a, 2) + 3 / 10 * tf.pow(a, 1)
    ap2 = (- (2 * x) / 7 - 1 / 5) * tf.pow(a, 3) + ((17 * x) / 7 + 3 / 10) * tf.pow(a, 2) + (
                9 / 10 - (22 * x) / 7) * tf.pow(a, 1) + x
    ap1 = tf.multiply(ap1, mask1)
    ap2 = tf.multiply(ap2, mask2)
    ap = tf.add(ap1, ap2)
    return ap

#this is for ablation study
def Ddz(a):
    res = tf.clip_by_value(a, 0.0, 1.0)
    return res

 # def metrics
acctrack = tf.keras.metrics.BinaryAccuracy()
losstrack = tf.keras.metrics.Mean(name='loss')
ptrack = tf.metrics.Precision(name='precision')
rtrack = tf.metrics.Recall(name='recall')
tptrack = tf.metrics.TruePositives(name='tp')
fptrack = tf.metrics.FalsePositives(name='fp')
tntrack = tf.metrics.TrueNegatives(name='tn')
fntrack = tf.metrics.FalseNegatives(name='fn')

class Fuzlayer(tf.keras.layers.Layer):

    def __init__(self, input_dim, time_len):
        super(Fuzlayer, self).__init__()
        self.input_dim = input_dim
        self.time_len = time_len

    def build(self, input_dim, time_len):
        #self.ramp = self.add_weight(name='fuz_w', shape=[input_dim, 1])
        #g1 = tf.random.Generator.from_seed(randomseed)
        '''
        self.ramp = tf.Variable(initial_value=tf.random.uniform(minval=1e-1, maxval=5e-1, shape=[input_dim, 1],
                                                                seed=randomseed, dtype=tf.float32,),
                                name='fuz_w',  trainable=True)'''
        self.ramp = tf.Variable(initial_value=tf.constant(1e-3, shape=[input_dim, 1], dtype=tf.float32,),
                                name='fuz_w', trainable=True)
        #self.bias = self.add_weight(initializer='zeros', name='fuz_b', shape=[input_dim, 1])
        self.built = True

    def call(self, inputs):
        output = tf.multiply(inputs, self.ramp)# + self.bias
        #output = tf.reshape(output, [lowfact * saptime, 1])
        output = tf.tanh(output)
        #1.建立一个镜像矩阵
        outputmir = output
        #2.建立正数的mask和负数的mask
        posmask = tf.where(output <= 0, x=0.0, y=1.0)
        negmask = tf.where(outputmir <= 0, x=-1.0, y=0.0)
        #3.mask掉不需要的阈值
        output = tf.multiply(posmask, output)
        outputmir = tf.multiply(negmask, outputmir)
        comp = tf.concat([output, outputmir], axis=0)
        #rescomp = tf.reshape(comp, [lowfact*2, saptime])
        #result = tf.reshape(rescomp, [lowfact*2, saptime])
        result = O2P(comp)
        #result = Ddz(comp)
        return result


class Graphlayer(tf.keras.layers.Layer):

    def __init__(self, output_dim, input_dim):
        super(Graphlayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, output_dim, input_dim):
        self.access = tf.Variable(graphconstruct.adj_ca(), dtype=tf.float32, trainable=False)
        self.graph_weights = tf.Variable(initial_value=graphconstruct.adj_causal_w(), trainable=True, name='gra_weight')
        self.built = True

    def call(self, inputv):
        weight = tf.multiply(self.graph_weights, tf.reduce_mean(inputv, axis=1))
        weight = normalize(weight)
        actiweight = tf.multiply(self.access, weight)
        actiweight = normalize(actiweight)
        out = tf.matmul(actiweight, inputv)
        out = O2P(out)
        return out


class FIGNN(tf.keras.models.Model):

    def __init__(self, itime):
        super(FIGNN, self).__init__()
        self.itime = itime
        self.fl = Fuzlayer(lowfact, saptime)
        self.gl = Graphlayer(highfact, 2*lowfact+highfact)
        self.fl.build(lowfact, saptime)
        self.gl.build(highfact, 2*lowfact+highfact)

    def call(self, data):
        xl = self.fl(data)
        x_res = tf.constant(eps, shape=[highfact, saptime])
        x_res = tf.concat([xl, x_res], axis=0)
        for i in range(0, self.itime):
            x_res = self.gl(x_res)
        x_res = tf.reduce_mean(x_res[208:208+21], axis=1)
        return x_res

    def train_step(self, datas, batch_size):
        loss = tf.constant(eps, shape=[21], dtype=tf.float32)
        with tf.GradientTape() as tape:
            dataset, labelset = datas
            for i in range(batch_size):
                data = dataset[i]
                label = labelset[i]
                y_pred = self(data)
                loss_l = self.loss_with_label(label, y_pred)
                #assign add 会带来梯度失去的问题
                loss = loss_l + loss
                acctrack.update_state(label, y_pred)
                ptrack.update_state(label, y_pred)
                rtrack.update_state(label, y_pred)
                tptrack.update_state(label, y_pred)
                tntrack.update_state(label, y_pred)
                fptrack.update_state(label, y_pred)
                fntrack.update_state(label, y_pred)
            lossm = tf.divide(loss, batch_size)
            grads = tape.gradient(lossm, [self.fl.ramp, self.gl.graph_weights, ])
            grads[0] = tf.multiply(grads[0], layer_ratio)
            #grads[1] = invnorm(self.gl.graph_weights, grads[1])
            self.optimizer.apply_gradients(zip(grads, [self.fl.ramp, self.gl.graph_weights, ]))
        losstrack.update_state(lossm)
        del loss, loss_l, data, label, dataset, labelset, y_pred
        gc.collect()
        return {m.name: m.result() for m in self.metrics}

    #@tf.function
    def test_step(self, x):
        data, label = x
        y_pred = self(data)
        #y_pred = y_pred[:, 1040:]
        y_pred = tf.transpose(y_pred)
        loss_l = self.loss_with_label(label, y_pred)
        loss = loss_l #+loss_k
        losstrack.update_state(loss)
        acctrack.update_state(label, y_pred)
        ptrack.update_state(label, y_pred)
        rtrack.update_state(label, y_pred)
        tptrack.update_state(label, y_pred)
        tntrack.update_state(label, y_pred)
        fptrack.update_state(label, y_pred)
        fntrack.update_state(label, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def loss_with_label(self, y_act, y_pred):
        #loss = tf.losses.binary_crossentropy(y_act, y_pred)
        label_loss = tf.nn.softmax_cross_entropy_with_logits(y_act, y_pred)
        regular_loss = tf.norm(self.fl.ramp, ord=2)
        loss = label_loss + 0e-4*regular_loss
        return loss

    def fidelityt(self, datas, noise):
        data, label = datas
        xl = self.fl(data)
        x_res = tf.constant(eps, shape=[highfact, saptime])
        x_res = tf.concat([xl, x_res], axis=0)
        for i in range(0, self.itime):
            x_res = self.gl(x_res)
        x_res = tf.reduce_mean(x_res[208:208 + 21], axis=1)
        data = data + (2* tf.random.uniform(shape=[50]) - 1)*noise
        xl1 = self.fl(data)
        x_res1 = tf.constant(eps, shape=[highfact, saptime])
        x_res1 = tf.concat([xl1, x_res1], axis=0)
        for i in range(0, self.itime):
            x_res1 = self.gl(x_res1)
        x_res1 = tf.reduce_mean(x_res1[208:208 + 21], axis=1)
        return tf.reduce_mean(x_res - x_res1).numpy()

    def fact_scaner(self, paramatrix, accessmatrix, datamatrix, index):
        para = paramatrix[index]
        access = accessmatrix[index]
        para = normalize(tf.multiply(access, normalize(para)))
        para = tf.expand_dims(para, axis=-1)
        res = tf.reduce_mean(tf.multiply(para, datamatrix), axis=1)
        index = tf.nn.top_k(res, k=min(2, len(res)), sorted=False).indices
        return index
    def interpreter(self, datainput, name, label):
        '''
        针对某一个样本的解释器
        '''

        rawdata = datainput
        lowfactindex = [
            'Feed flow component A (stream 1)_Pos',
            'Feed flow component D (stream 2)_Pos',
            'Feed flow component E (stream 3)_Pos',
            'Feed flow components A & C (stream 4)_Pos',
            'Recycle flow to reactor from separator (stream 8)_Pos',
            'Reactor feed (stream 6)_Pos',
            'Reactor pressure_Pos',
            'Reactor level_Pos',
            'Reactor temperature_Pos',
            'Purge flow (stream 9)_Pos',
            'Separator temperature_Pos',
            'Separator level_Pos',
            'Separator pressure_Pos',
            'Sperator underflow (liquid phase)_Pos',
            'Stripper level_Pos',
            'Stripper pressure_Pos',
            'Stripper underflow (stream 11)_Pos',
            'Stripper temperature_Pos',
            'Stripper steam flow_Pos',
            'Compressor work_Pos',
            'Reactor cooling water outlet temperature_Pos',
            'Condenser cooling water outlet temperature_Pos',
            'Concentration of A in Reactor feed (stream 6)_Pos',
            'Concentration of B in Reactor feed (stream 6)_Pos',
            'Concentration of C in Reactor feed (stream 6)_Pos',
            'Concentration of D in Reactor feed (stream 6)_Pos',
            'Concentration of E in Reactor feed (stream 6)_Pos',
            'Concentration of F in Reactor feed (stream 6)_Pos',
            'Concentration of A in Purge (stream 9)_Pos',
            'Concentration of B in Purge (stream 9)_Pos',
            'Concentration of C in Purge (stream 9)_Pos',
            'Concentration of D in Purge (stream 9)_Pos',
            'Concentration of E in Purge (stream 9)_Pos',
            'Concentration of F in Purge (stream 9)_Pos',
            'Concentration of G in Purge (stream 9)_Pos',
            'Concentration of H in Purge (stream 9)_Pos',
            'Concentration of D in stripper underflow (stream 11)_Pos',
            'Concentration of E in stripper underflow (stream 11)_Pos',
            'Concentration of F in stripper underflow (stream 11)_Pos',
            'Concentration of G in stripper underflow (stream 11)_Pos',
            'Concentration of H in stripper underflow (stream 11)_Pos',
            'Valve position feed component D (stream 2)_Pos',
            'Valve position feed component E (stream 3)_Pos',
            'Valve position feed component A (stream 1)_Pos',
            'Valve position feed component A & C (stream 4)_Pos',
            'Valve position compressor re-cycle_Pos',
            'Valve position purge (stream 9)_Pos',
            'Valve position underflow separator (stream 10)_Pos',
            'Valve position underflow stripper (stream 11)_Pos',
            'Valve position stripper steam_Pos',
            'Valve position cooling water outlet of reactor_Pos',
            'Valve position cooling water outlet of separator_Pos',
            'Feed flow component A (stream 1)_D_Pos',
            'Feed flow component D (stream 2)_D_Pos',
            'Feed flow component E (stream 3)_D_Pos',
            'Feed flow components A & C (stream 4)_D_Pos',
            'Recycle flow to reactor from separator (stream 8)_D_Pos',
            'Reactor feed (stream 6)_D_Pos',
            'Reactor pressure_D_Pos',
            'Reactor level_D_Pos',
            'Reactor temperature_D_Pos',
            'Purge flow (stream 9)_D_Pos',
            'Separator temperature_D_Pos',
            'Separator level_D_Pos',
            'Separator pressure_D_Pos',
            'Sperator underflow (liquid phase)_D_Pos',
            'Stripper level_D_Pos',
            'Stripper pressure_D_Pos',
            'Stripper underflow (stream 11)_D_Pos',
            'Stripper temperature_D_Pos',
            'Stripper steam flow_D_Pos',
            'Compressor work_D_Pos',
            'Reactor cooling water outlet temperature_D_Pos',
            'Condenser cooling water outlet temperature_D_Pos',
            'Concentration of A in Reactor feed (stream 6)_D_Pos',
            'Concentration of B in Reactor feed (stream 6)_D_Pos',
            'Concentration of C in Reactor feed (stream 6)_D_Pos',
            'Concentration of D in Reactor feed (stream 6)_D_Pos',
            'Concentration of E in Reactor feed (stream 6)_D_Pos',
            'Concentration of F in Reactor feed (stream 6)_D_Pos',
            'Concentration of A in Purge (stream 9)_D_Pos',
            'Concentration of B in Purge (stream 9)_D_Pos',
            'Concentration of C in Purge (stream 9)_D_Pos',
            'Concentration of D in Purge (stream 9)_D_Pos',
            'Concentration of E in Purge (stream 9)_D_Pos',
            'Concentration of F in Purge (stream 9)_D_Pos',
            'Concentration of G in Purge (stream 9)_D_Pos',
            'Concentration of H in Purge (stream 9)_D_Pos',
            'Concentration of D in stripper underflow (stream 11)_D_Pos',
            'Concentration of E in stripper underflow (stream 11)_D_Pos',
            'Concentration of F in stripper underflow (stream 11)_D_Pos',
            'Concentration of G in stripper underflow (stream 11)_D_Pos',
            'Concentration of H in strip per underflow (stream 11)_D_Pos',
            'Valve position feed component D (stream 2)_D_Pos',
            'Valve position feed component E (stream 3)_D_Pos',
            'Valve position feed component A (stream 1)_D_Pos',
            'Valve position feed component A & C (stream 4)_D_Pos',
            'Valve position compressor re-cycle_D_Pos',
            'Valve position purge (stream 9)_D_Pos',
            'Valve position underflow separator (stream 10)_D_Pos',
            'Valve position underflow stripper (stream 11)_D_Pos',
            'Valve position stripper steam_D_Pos',
            'Valve position cooling water outlet of reactor_D_Pos',
            'Valve position cooling water outlet of separator_D_Pos',
            'Feed flow component A (stream 1)_Neg',
            'Feed flow component D (stream 2)_Neg',
            'Feed flow component E (stream 3)_Neg',
            'Feed flow components A & C (stream 4)_Neg',
            'Recycle flow to reactor from separator (stream 8)_Neg',
            'Reactor feed (stream 6)_Neg',
            'Reactor pressure_Neg',
            'Reactor level_Neg',
            'Reactor temperature_Neg',
            'Purge flow (stream 9)_Neg',
            'Separator temperature_Neg',
            'Separator level_Neg',
            'Separator pressure_Neg',
            'Sperator underflow (liquid phase)_Neg',
            'Stripper level_Neg',
            'Stripper pressure_Neg',
            'Stripper underflow (stream 11)_Neg',
            'Stripper temperature_Neg',
            'Stripper steam flow_Neg',
            'Compressor work_Neg',
            'Reactor cooling water outlet temperature_Neg',
            'Condenser cooling water outlet temperature_Neg',
            'Concentration of A in Reactor feed (stream 6)_Neg',
            'Concentration of B in Reactor feed (stream 6)_Neg',
            'Concentration of C in Reactor feed (stream 6)_Neg',
            'Concentration of D in Reactor feed (stream 6)_Neg',
            'Concentration of E in Reactor feed (stream 6)_Neg',
            'Concentration of F in Reactor feed (stream 6)_Neg',
            'Concentration of A in Purge (stream 9)_Neg',
            'Concentration of B in Purge (stream 9)_Neg',
            'Concentration of C in Purge (stream 9)_Neg',
            'Concentration of D in Purge (stream 9)_Neg',
            'Concentration of E in Purge (stream 9)_Neg',
            'Concentration of F in Purge (stream 9)_Neg',
            'Concentration of G in Purge (stream 9)_Neg',
            'Concentration of H in Purge (stream 9)_Neg',
            'Concentration of D in stripper underflow (stream 11)_Neg',
            'Concentration of E in stripper underflow (stream 11)_Neg',
            'Concentration of F in stripper underflow (stream 11)_Neg',
            'Concentration of G in stripper underflow (stream 11)_Neg',
            'Concentration of H in stripper underflow (stream 11)_Neg',
            'Valve position feed component D (stream 2)_Neg',
            'Valve position feed component E (stream 3)_Neg',
            'Valve position feed component A (stream 1)_Neg',
            'Valve position feed component A & C (stream 4)_Neg',
            'Valve position compressor re-cycle_Neg',
            'Valve position purge (stream 9)_Neg',
            'Valve position underflow separator (stream 10)_Neg',
            'Valve position underflow stripper (stream 11)_Neg',
            'Valve position stripper steam_Neg',
            'Valve position cooling water outlet of reactor_Neg',
            'Valve position cooling water outlet of separator_Neg',
            'Feed flow component A (stream 1)_D_Neg',
            'Feed flow component D (stream 2)_D_Neg',
            'Feed flow component E (stream 3)_D_Neg',
            'Feed flow components A & C (stream 4)_D_Neg',
            'Recycle flow to reactor from separator (stream 8)_D_Neg',
            'Reactor feed (stream 6)_D_Neg',
            'Reactor pressure_D_Neg',
            'Reactor level_D_Neg',
            'Reactor temperature_D_Neg',
            'Purge flow (stream 9)_D_Neg',
            'Separator temperature_D_Neg',
            'Separator level_D_Neg',
            'Separator pressure_D_Neg',
            'Sperator underflow (liquid phase)_D_Neg',
            'Stripper level_D_Neg',
            'Stripper pressure_D_Neg',
            'Stripper underflow (stream 11)_D_Neg',
            'Stripper temperature_D_Neg',
            'Stripper steam flow_D_Neg',
            'Compressor work_D_Neg',
            'Reactor cooling water outlet temperature_D_Neg',
            'Condenser cooling water outlet temperature_D_Neg',
            'Concentration of A in Reactor feed (stream 6)_D_Neg',
            'Concentration of B in Reactor feed (stream 6)_D_Neg',
            'Concentration of C in Reactor feed (stream 6)_D_Neg',
            'Concentration of D in Reactor feed (stream 6)_D_Neg',
            'Concentration of E in Reactor feed (stream 6)_D_Neg',
            'Concentration of F in Reactor feed (stream 6)_D_Neg',
            'Concentration of A in Purge (stream 9)_D_Neg',
            'Concentration of B in Purge (stream 9)_D_Neg',
            'Concentration of C in Purge (stream 9)_D_Neg',
            'Concentration of D in Purge (stream 9)_D_Neg',
            'Concentration of E in Purge (stream 9)_D_Neg',
            'Concentration of F in Purge (stream 9)_D_Neg',
            'Concentration of G in Purge (stream 9)_D_Neg',
            'Concentration of H in Purge (stream 9)_D_Neg',
            'Concentration of D in stripper underflow (stream 11)_D_Neg',
            'Concentration of E in stripper underflow (stream 11)_D_Neg',
            'Concentration of F in stripper underflow (stream 11)_D_Neg',
            'Concentration of G in stripper underflow (stream 11)_D_Neg',
            'Concentration of H in stripper underflow (stream 11)_D_Neg',
            'Valve position feed component D (stream 2)_D_Neg',
            'Valve position feed component E (stream 3)_D_Neg',
            'Valve position feed component A (stream 1)_D_Neg',
            'Valve position feed component A & C (stream 4)_D_Neg',
            'Valve position compressor re-cycle_D_Neg',
            'Valve position purge (stream 9)_D_Neg',
            'Valve position underflow separator (stream 10)_D_Neg',
            'Valve position underflow stripper (stream 11)_D_Neg',
            'Valve position stripper steam_D_Neg',
            'Valve position cooling water outlet of reactor_D_Neg',
            'Valve position cooling water outlet of separator_D_Neg',
             'Fault 0', 'Fault 1', 'Fault 2', 'Fault 3', 'Fault 4', 'Fault 5', 'Fault 6', 'Fault 7',
            'Fault 8', 'Fault 9', 'Fault 10', 'Fault 11', 'Fault 12', 'Fault 13', 'Fault 14', 'Fault 15', 'Fault 16',
            'Fault 17', 'Fault 18', 'Fault 19', 'Fault 20',
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
            '31', '32', '33', '34', '35', '36', '37', '38', '39',
                        ]
        fuzzydata = self.fl(rawdata)
        fuzzydata = tf.concat([fuzzydata, tf.zeros([highfact, saptime])], axis=0)
        l1data = self.gl(fuzzydata)
        l2data = self.gl(l1data)
        l3data = self.gl(l2data)
        l4data = self.gl(l3data)
        x_res = tf.nn.top_k(tf.reduce_mean(l4data[208:208 + 21], axis=1), k=1).indices
        x_res = np.squeeze(x_res.numpy())
        dot = gz.Digraph('4类'+str(label)+'分类结果：' + str(x_res) +'_sap'+ name,
                         'comment',None,None,'png',None,"UTF-8",
       {'rankdir':'TB'},
       {'color':'black','fontcolor':'black','fontname':'FangSong','fontsize':'12','style':'rounded','shape':'box'},
       {'color':'#999999','fontcolor':'#888888','fontsize':'10','fontname':'FangSong'},None,True)
        dot.graph_attr['dpi'] = '600'
        dot.node(str(x_res), lowfactindex[x_res+208])
        index1 = self.fact_scaner(self.gl.graph_weights, self.gl.access, l3data, x_res+208)
        cache0 = str(x_res)
        for item in index1:
            index2 = self.fact_scaner(self.gl.graph_weights, self.gl.access, l2data, item)
            dot.node(str(item), str((item % 104 + 1).numpy()) + lowfactindex[item], color='navyblue')
            dot.edge(str(item), str(cache0), label='L3', color='navyblue')
            cache = str(item)
            if isinstance(index2, int):
                continue
            for item in index2:
                index3 = self.fact_scaner(self.gl.graph_weights, self.gl.access, l1data, item)
                dot.node(str(item), str((item % 104 + 1).numpy()) + lowfactindex[item], color='cyan3')
                dot.edge(str(item), cache, label='L2', color='cyan3')
                cache2 = str(item)
                if isinstance(index3, int):
                    continue
                for item in index3:
                    dot.node(str(item), str((item % 104 + 1).numpy()) + lowfactindex[item], color='green3')
                    dot.edge(str(item), cache2, label='L1', color='green3')
                    index4 = self.fact_scaner(self.gl.graph_weights, self.gl.access, fuzzydata, item)
                    cache3 = str(item)
                    for item in index4:
                        dot.node(str(item), str((item % 104 + 1).numpy()) + lowfactindex[item], color='orange3')
                        dot.edge(str(item), cache3, label='L0', color='orange3')

        dot.render(directory='./figs/corr')
        return

    def interpreter_analysis(self, datainput):
        rawdata = datainput
        array = np.zeros(shape=[4,248])
        fuzzydata = self.fl(rawdata)
        fuzzydata = tf.concat([fuzzydata, tf.zeros([highfact, saptime])], axis=0)
        l1data = self.gl(fuzzydata)
        l2data = self.gl(l1data)
        l3data = self.gl(l2data)
        l4data = self.gl(l3data)
        x_res = tf.nn.top_k(tf.reduce_mean(l4data[208:208 + 21], axis=1), k=1).indices
        x_res = np.squeeze(x_res.numpy())
        index1 = self.fact_scaner(self.gl.graph_weights, self.gl.access, l3data, x_res + 208)
        for item in index1:
            array[0, item] = array[0, item] + 1
            index2 = self.fact_scaner(self.gl.graph_weights, self.gl.access, l2data, item)
            if isinstance(index2, int):
                continue
            for item in index2:
                array[1, item] = array[1, item] + 1
                index3 = self.fact_scaner(self.gl.graph_weights, self.gl.access, l1data, item)
                for item in index3:
                    array[2, item] = array[2, item] + 1
                    index4 = self.fact_scaner(self.gl.graph_weights, self.gl.access, fuzzydata, item)
                    for item in index4:
                        array[3, item] = array[3, item] + 1

        return array
    def call_pattern(self, data):
        xl = self.fl(data)
        x_res = tf.constant(eps, shape=[highfact, saptime])
        x_res = tf.concat([xl, x_res], axis=0)
        for i in range(0, self.itime-1):
            x_res = self.gl(x_res)
        x_res = tf.reduce_mean(x_res[0:208], axis=1)
        return x_res

    def gradcaculate(self, datas, batch_size):
        loss = tf.constant(eps, shape=[21], dtype=tf.float32)
        with tf.GradientTape() as tape:
            dataset, labelset = datas
            for i in range(batch_size):
                data = dataset[i]
                label = labelset[i]
                y_pred = self(data)
                loss_l = self.loss_with_label(label, y_pred)
                #assign add 会带来梯度失去的问题
                loss = loss_l + loss
            lossm = tf.divide(loss, batch_size)
            grads = tape.gradient(lossm, [self.fl.ramp, self.gl.graph_weights])
        return grads[1]
    @property
    def metrics(self):
        return [losstrack, acctrack, ptrack, rtrack, tptrack, fptrack, tntrack, fntrack]