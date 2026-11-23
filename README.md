# Handwriting recognition

Reads lines of handwritten English text. Images of full text lines from the IAM dataset are
normalised to a fixed height, and a CNN + bidirectional LSTM predicts the character sequence.

## Pipeline

1. **Data** (`Preprocessing.ipynb`): the [IAM-line](https://huggingface.co/datasets/Teklia/IAM-line)
   dataset from Hugging Face (parquet). Images are decoded, resized to a height of 128 pixels while
   keeping the aspect ratio, padded to 128x2000 and scaled to [0, 1].
2. **Labels**: each line of text is encoded over a 63-symbol alphabet (letters, digits, space) and
   padded to 80 characters.
3. **Model** (`train.ipynb`): convolution and pooling blocks with batch norm and dropout reduce the image
   to a feature sequence; two bidirectional LSTM layers read it, and a dense softmax predicts one
   character per step (about 7M parameters).
4. **Weights**: `trial/` (SavedModel) and `trial.h5` hold a trained checkpoint.

## Run

```bash
pip install -r requirements.txt tensorflow jupyter
jupyter lab
```

## Stack
Python · TensorFlow / Keras · Pillow · pandas / pyarrow

## Licence
Apache 2.0. IAM data is subject to its own terms of use.
