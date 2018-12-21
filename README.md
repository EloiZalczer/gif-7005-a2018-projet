# gif-7005-a2018-projet

[Dataset du projet](https://research.google.com/audioset/download.html)

##usage

#h6 download n first videos from dataset (you can specify classes if the argument is empty it will dl anything):
$ ipython
$ run yt_to_img.py
$ downloadClass(classLabel = [], n=10)

#h6 train and save classifier 
$ ipython
$ run a2018-projet/main.py -e 10 -i data/bal_train.h5 -t data/eval.h5 -s True

#h6 extract vggish output for resNet classifierfrom a .wav file 
$ ipython
$ run yt_to_img.py
$ example = vggExamples(filename = 'blob.wav')

#predict classes for the extracted caracteristics
$ ipython
$ run a2018-projet/main.py -e 10 -i data/bal_train.h5 -t data/eval.h5 -l True
$ model.predictSingle(example[0][0])
