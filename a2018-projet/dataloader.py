#!/usr/bin/python

from os import listdir, sys, path
import tensorflow as tf

class DataLoader:
    def __init__(self, filename):
        self.filepath = path.abspath(filename)
        self.files = [path.join(self.filepath, f) for f in listdir(self.filepath) if path.isfile(path.join(self.filepath, f))]
        if len(self.files) == 0:
            print("No data found in given directory")
            sys.exit(2)

    def load_data(self):
        dataset = tf.data.TFRecordDataset(self.files)
        dataset = dataset.batch(32)
        dataset = dataset.repeat()

        # preprocess(dataset)

        return dataset