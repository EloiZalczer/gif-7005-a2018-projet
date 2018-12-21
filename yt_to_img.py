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
import sys
sys.path.insert(0, './models/research/audioset/')
from vggish_postprocess import Postprocessor
import tensorflow as tf
import vggish_slim

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
        subprocess.call(['ffmpeg', '-i', 'sound.wav', '-ss', start_time, '-to', end_time, '-acodec', 'pcm_s16le', '_'+url+'.wav'])
    
    subprocess.call(['rm', 'sound.wav'])

def cleanDir():
    subprocess.call(['rm', '*.wav'])

def downloadClass(classLabels = [], n = 100):
    count = 0
    with open('balanced_train_segments.csv', 'r', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar="'")
        for row in spamreader:
            count += 1
            if (count >= 5):
                if (classLabels == None and count < n):
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

def vggExamples(fileName = None):
    if(fileName == None):
        files = list(filter(lambda x: x.endswith('.wav'),os.listdir()))
    else:
        files = [fileName] 

    PCAfiles = list()
    for f in files:
        subprocess.call(['python3', 'models/research/audioset/custom_inference.py', f])
        PCAexample = np.load("./PCA.npy")
        PCAfiles.append(PCAexample)

    return PCAfiles
