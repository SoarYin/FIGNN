import tensorflow as tf


def normalize(tensor):
    eps = 1e-8
    tensor = tf.transpose(tensor)
    max = tf.reduce_max(tensor, axis=0)
    min = tf.reduce_min(tensor, axis=0)
    output = tf.subtract(tensor, min - eps)
    output = tf.divide(output, max-min+eps)
    nor = tf.norm(output, 1, axis=0)
    output = tf.divide(output, nor)
    output = tf.transpose(output)
    return output

def invnorm(tensor, transtensor):
    eps = 1e-8
    tensor = tf.transpose(tensor)
    transtensor = tf.transpose(transtensor)
    max = tf.reduce_max(tensor, axis=0)
    min = tf.reduce_min(tensor, axis=0)
    inside = tf.subtract(tensor, min - eps)
    inside = tf.divide(inside, max - min + eps)
    norm = tf.norm(inside, 1, axis=0)
    output = tf.multiply(transtensor, norm)
    output = tf.multiply(output, max - min + eps)
    output = tf.add(output, min)
    output = tf.transpose(output)
    return output
