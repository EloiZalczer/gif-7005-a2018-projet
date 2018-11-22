#!/usr/bin/python

import tensorflow as tf
from tensorflow.data import TFRecordDataset
from os import listdir, sys, path
import tables
import numpy as np

filepath = "/home/eloi/Documents/Laval/Machine_learning/Projet/gif-7005-a2018-projet/data/audioset_v1_embeddings/bal_train/"
files = [path.join(filepath, f) for f in listdir(filepath) if path.isfile(path.join(filepath, f))]

hdf5_path = "/home/eloi/Documents/Laval/Machine_learning/Projet/gif-7005-a2018-projet/convert2.hdf5"

context_features = {'labels': tf.VarLenFeature(tf.int64)}
sequence_features = {'audio_embedding': tf.VarLenFeature(tf.string)}

print("Sequence features : ", sequence_features)

hdf5_file = tables.open_file(hdf5_path, mode='w')

def _parse_function(example_proto):
    print("example_proto : ", example_proto)
    contexts, features = tf.parse_single_sequence_example(
        example_proto,
        context_features=context_features,
        sequence_features=sequence_features)

    return contexts, features

dataset = TFRecordDataset(files)

dataset = dataset.map(_parse_function)

iterator = dataset.make_one_shot_iterator()

data_shape=(0,)
labels_shape=(0,)

sound_dtype = tables.StringAtom(itemsize=128)
labels_dtype = tables.IntAtom()

data_storage = hdf5_file.create_earray(hdf5_file.root, 'audio_embedding', sound_dtype, shape=data_shape)
labels_storage = hdf5_file.create_earray(hdf5_file.root, 'labels', labels_dtype, shape=labels_shape)

value = iterator.get_next()
i = 1
with tf.Session() as sess:
    while 1:
        try:
            tmp = sess.run(value)
            labels = tmp[0]['labels'].values
            audio_embeddings = tmp[1]['audio_embedding'].values
            # print("Labels : ", labels)
            # print("Audio embeddings : ", audio_embeddings)
            data_storage.append(audio_embeddings)
            labels_storage.append(labels)
            print("Enregistrement ", i)
            i+=1
        except tf.errors.OutOfRangeError:
            break

hdf5_file.close()