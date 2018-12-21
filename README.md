# gif-7005-a2018-projet

[Dataset du projet](https://research.google.com/audioset/download.html)

## Usage

# Installation of Vggish

`cd models/research/audioset`
# Download data files into same directory as code.
`curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt`
`curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz`

# Installation ready, let's test it.
`python vggish_smoke_test.py`
# If we see "Looks Good To Me", then we're all set.

# Download n first videos from dataset (you can specify classes if the argument is empty it will dl anything):
`ipython`
`run yt_to_img.py`
`downloadClass(classLabel = [], n=10)`

# Train and save classifier 
`ipython`
`run a2018-projet/main.py -e 10 -i data/bal_train.h5 -t data/eval.h5 -s True`

# Extract vggish output for resNet classifierfrom a .wav file 
`ipython`
`run yt_to_img.py`
`example = vggExamples(filename = 'blob.wav')`

# Predict classes for the extracted caracteristics
`ipython`
`run a2018-projet/main.py -e 10 -i data/bal_train.h5 -t data/eval.h5 -l True`
`model.predictSingle(example[0][0])`

