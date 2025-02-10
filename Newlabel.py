import numpy as np
import tensorflow as tf

def Newlabel(label,id):
    emptylable = np.zeros((21), dtype=float)
    if id != 0:
        emptylable[id] = 1
    label = emptylable
    label = tf.convert_to_tensor(label, dtype=float)
    return label

def Newlabelt(label,id):
    emptylable = np.zeros((21), dtype=float)
    if id != 0:
        emptylable[id] = 1
    label = emptylable
    label = tf.convert_to_tensor(label, dtype=float)
    return label

