#!/usr/bin/python

import sys, getopt

import numpy as np
import matplotlib.pyplot as plt

from dataloader import DataLoader
from model import FullyConnectedNet

import torch
import torch.nn as nn
from torch.optim import SGD

class SoundRecognition():
    def __init__(self):
        self.verbose = False
        self.idir=""
        self.testdir=""
        self.epochs = 1
        self.device = 'cpu'

    def help(self):
        print("Usage : main.py [-hv] [-e <epochs>] -i <input file.h5> -t <test file.h5>")

    def load_args(self, args):
        try:
            opts, args = getopt.getopt(args, "e:hi:t:v", ["help", "idir="])
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
            elif o in ("-t", "--test"):
                self.testdir = a
            elif o in ("-e", "--epochs"):
                self.epochs = a
            else:
                assert False, "Unhandled option"

        if self.idir == "":
            self.help()
            sys.exit()

        if self.testdir == "":
            self.help()
            sys.exit()

        print(self.idir)

    def train(self):

        self.Model.to(self.device)
        criterion = nn.BCELoss()
        optimizer = SGD(self.Model.parameters(), lr=0.01, momentum=0.9)
        self.Model.train()

        for i_epoch in range(self.epochs):
            print("Running one epoch")
            for i_batch, batch in enumerate(self.train_dataset):
                audio_embedding, labels = batch

                #print("audio_embedding : ", audio_embedding)

                audio_embedding = audio_embedding.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                predictions = self.Model(audio_embedding)
                print("Batch number ", i_batch)
                loss = criterion(predictions, labels)

                loss.backward()
                optimizer.step()

    def compute_accuracy(self):
        self.Model.eval()

        all_predictions = []
        all_targets = []
        for i_batch, batch in enumerate(self.test_dataset):
            audio_embedding, labels = batch

            print("Predicting batch ", i_batch)

            audio_embedding = audio_embedding.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                predictions = self.Model(audio_embedding)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        predictions_numpy = np.concatenate(all_predictions, axis=0)
        predictions_numpy[predictions_numpy>=0.5] = 1.0
        predictions_numpy[predictions_numpy<0.5] = 0.0
        y = np.zeros(shape=predictions_numpy.shape)
        targets_numpy = np.concatenate(all_targets, axis=0)
        predictions_numpy = np.where(targets_numpy, predictions_numpy, y)
        return len(np.where((np.any(predictions_numpy == 1, axis=1)) == True)[0])/len(predictions_numpy)

    def run(self, args):
        self.load_args(args)
        dataloader_args = {'batch_size': 32,
                           'num_workers': 0,
                           'shuffle': True}

        trainDataLoader = DataLoader(self.idir, **dataloader_args)
        self.train_dataset = trainDataLoader.load_data()

        testDataLoader = DataLoader(self.testdir, **dataloader_args)
        self.test_dataset = testDataLoader.load_data()

        self.Model = FullyConnectedNet()

        self.train()

        accuracy = self.compute_accuracy()

        print("Accuracy : ", accuracy)



if __name__ == '__main__':
    model = SoundRecognition()
    model.run(sys.argv[1:])