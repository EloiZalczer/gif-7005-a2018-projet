# gif-7005-a2018-projet

[Dataset du projet](https://research.google.com/audioset/download.html)

## Usage

# Installation of Vggish

`cd models/research/audioset`<br />
# Download data files into same directory as code.
`curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt`<br />
`curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz`<br />

# Installation ready, let's test it.
`python vggish_smoke_test.py`<br />
# If we see "Looks Good To Me", then we're all set.

# Download n first videos from dataset (you can specify classes if the argument is empty it will dl anything):
`ipython`<br />
`run yt_to_img.py`<br />
`downloadClass(classLabel = [], n=10)`<br />

# Train and save classifier 
`ipython`<br />
`run a2018-projet/main.py -e 10 -i data/bal_train.h5 -t data/eval.h5 -s True`<br />

# Extract vggish output for resNet classifierfrom a .wav file 
`ipython`<br />
`run yt_to_img.py`<br />
`example = vggExamples(filename = 'blob.wav')`<br />

# Predict classes for the extracted caracteristics
`ipython`<br />
`run a2018-projet/main.py -e 10 -i data/bal_train.h5 -t data/eval.h5 -l True`<br />
`model.predictSingle(example[0][0])`<br />

