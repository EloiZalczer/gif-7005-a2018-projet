from __future__ import unicode_literals
import youtube_dl
import subprocess
import numpy as np
import sys
sys.path.insert(0, './models/research/audioset/')
from vggish_input import wavfile_to_examples
import os
import csv
import json

def download_sample(url, start_time, end_time):
    ydl_opts = {
        'outtmpl': 'sound.wav', 
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    print(start_time, end_time)
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['https://www.youtube.com/watch?v='+url+'&feature=youtu.be'])
        subprocess.call(['ffmpeg', '-i', 'sound.wav', '-ss', start_time, '-to', end_time, '-acodec', 'pcm_s16le', url+'.wav'])
    subprocess.call(['rm', 'sound.wav'])


def cleanDir():
    subprocess.call(['rm', '*.wav'])

def downloadClass(classLabels = []):
    count = 0
    with open('balanced_train_segments.csv', 'r', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar="'")
        for row in spamreader:
            count += 1
            if (count >= 5):
                if (classLabels == None):
                    try:
                        download_sample(row[0], row[1], row[2]) 
                    except Exception as inst:
                        print(inst)
                        
                else:
                    string = ','.join(row[3:])
                    labels = string[2:-1].split(',')
                    for l in labels:
                        if(l in classLabels):
                            try:
                                download_sample(row[0], row[1], row[2]) 
                            except Exception as inst:
                                print(inst)

'''
count = 0

subprocess.call(['rm', '*.wav'])

with open('balanced_train_segments.csv', 'r', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        count += 1
        if (count >= 5):
            #print(', '.join(row))
            try:
                download_sample(row[0], row[1], row[2]) 
            except Exception as inst:
                print(inst)
            if(count >= 105):
                break


files = list(filter(lambda s: s[-4:] == '.wav',os.listdir()))

X = np.zeros((1,96,64))

for i in range(len(files)):
    X = np.concatenate((X, wavfile_to_examples(files[i])), axis = 0)
    
X = X[1:,:,:]

from skimage import io

for x in X:
    print(x.shape)
    io.imshow(x)
    io.show()
''' 
                            