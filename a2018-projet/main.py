#!/usr/bin/python

import sys, getopt
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from dataloader import DataLoader

class SoundRecognition():
    def __init__(self):
        self.verbose = False
        self.idir=""

    def help(self):
        print("Usage : main.py [-h] -i <input directory>")

    def load_args(self, args):
        try:
            opts, args = getopt.getopt(args, "hi:v", ["help", "idir="])
        except getopt.GetoptError as err:
            print(str(err))
            self.help()
            sys.exit(2)

        for o, a in opts:
            if o == "-v":
                self.verbose = True
            elif o in ("-h", "--help"):
                self.help()
                sys.exit()
            elif o in ("-i", "--input"):
                self.idir = a
            else:
                assert False, "Unhandled option"

    def train(self):

        # model = tf.keras.Model(...)

        pass

    def run(self, args):
        self.load_args(args)
        dataLoader = DataLoader(self.idir)
        self.dataset = dataLoader.load_data()

        config = tf.ConfigProto()
        sess = tf.Session(config=config)



def main(args):
    model = SoundRecognition()
    model.run(args)

if __name__ == '__main__':
    tf.app.run(main=main, argv=sys.argv[1:])
