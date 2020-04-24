# Anonymous Code Base

Codebase for our submussion to EMNLP2020 Confluence

## Provided Models
In this section, we present several time-series approaches to
model valence ratings on the SENDv1. We implement:

### Self-attention Encoder + LSTM Decoder
This model is for one of experiment running on Stanford Emotional Narratives Dataset (SEND).

### Self-attention Encoder + Multi-layer Perception Decoder 
This model is for one of experiment running on Stanford Sentiment Treebank Dataset (SST).

## Requirement

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the required packages mentioned in the ***requirement.txt***.

```bash
pip install -r requirements.txt
```

## Usage

You will first need to download all the datasets needed and put it under the desired folder in a designed format. For SEND dataset, you can download from [SEND Repo](https://github.com/StanfordSocialNeuroscienceLab/SEND). For SST dataset, you can download from a repository we provided as SST is pretty small from [SST Ready-to-use Repo](https://github.com/frankaging/SST2-Sentence). Then, you need to place the datasets under a desired directory. You may change where you save the dataset in our code. You will also need Warriner dictionary place under a desired directory, and you may need to change code accordingly.

For all running on SEND,
```python
cd code/model
python train.py --dataset SEND
```

For all running on SST,
```python
cd code/model
python train.py --dataset SST
```

## Experiment You Can Try

After you train you model, you can place your model under save_model directory within a folder with a UUID encripted name. We provide 4 different experimental scripts you can play with.

For calculating the token level attentions (must run this first),
```python
cd code/model
python attn_analyze.py --dataset [SEND or SST]
```

For generating word cloud based on attentions,
```python
cd code/model
python word_cloud.py --dataset [SEND or SST]
```

For generating word cloud based on attentions,
```python
cd code/model
python sentence_highlight.py
```

For generating visualization of transformer's attention,
```python
cd code/model
python attention_viz.py --dataset [SEND or SST]
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
