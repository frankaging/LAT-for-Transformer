# The Transformer Attends to Semantics: A General Method for Backing Out Meaningful Self-attention Scores
Codebase for our submussion to EMNLP2020 Confluence

## Description
In this paper, we propose a Layer-wise Attention Tracing (LAT) method for backing out the attention score assigned to input tokens within a self-attention architecture.
We apply LAT to Transformer models trained on two tasks that have surface dissimilarities, but share common semantics: sentiment analysis of movie reviews and valence prediction in life story narratives. Across both tasks, the highly-attended words picked out by LAT tend to be rich in emotional semantics, as validated by an emotion lexicon.

## Models and Datasets

### Self-attention Encoder + LSTM Decoder
This model is for one of experiment running on Stanford Emotional Narratives Dataset (SEND).

### Self-attention Encoder + Multi-layer Perception Decoder 
This model is for one of experiment running on Stanford Sentiment Treebank Dataset (SST).

### Datasets
You will first need to download all the datasets needed and put it under the desired folder in a designed format. Then, you need to place the datasets under a desired directory. You may change where you save the dataset in our code. You will also need Warriner dictionary place under a desired directory, and you may need to change code accordingly.

## How to reproduce all of our results:

### Data Usage
All of the datasets can be downloaded from the links above. But since the dataset is not large, we are providing a simple downloadable version as well. After you download, you can put whereever you want as long as you specify your path while analyzing. However, we do recommend you put in the same directory of this code folder, and do not change the folder name.

### Requirements
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the required packages mentioned in the ***requirement.txt***.
```bash
pip install -r requirements.txt
```

### Populate Required Datasets
#### SEND
You will have to unzip the file, and then go the folder and run the populate command as follows. Do not change the directories in the folder.
```bash
python populate.py
```

#### SST
You will have to unzip the file, and then go the folder and run the populate command as follows. Do not change the directories in the folder.
```bash
python preprocess.py
```

### Training (Not Required)
**We provide pre-train models you can play with.** In case you want to retrain the model, you can use the following command
```python
cd code/model
python train.py `--`dataset [SEND or SST] `--`data_dir [path_to_data_folder] `--`model_dir [path_to_save_model] `--`unit_test False
```

## Experiments You Can Try
After you train you model, We provide different scripts of experiments you can play with, which produce results we show in the paper. Before you play with the experiments, note that if you retrain your model, in order to get what **your models** will generate, you will need to run the following script to overwrite saved results from **pretrained models**. Note that this is completely optional as we provide all the pretrained models and pre-extracted weights needed for you to demo these experiments.

### (No Retrain) Python Notebook Illustration
You can directly load with jupyter notebook installed. You can open in browser for demo.
```python
cd code/model
jupyter notebook
```

#### attention_viz
This scripts will generate the flow diagram shown in the paper. It provides helper functions that can easily visualize attention flows with in the attention network, you can refer to the `attention_util.py` for details.

#### head_viz
This scripts will generate the head attention heatmap.

### (Only In Case Retrain) Re-extract All Attention Weights
Step 1: (Required) Re-extract all needed weights.
```python
python attn_analyze.py `--`dataset [SEND or SST] `--`data_dir [path_to_data_folder] `--`model_path [path_to_save_model] `--`out_dir [path_to_save_result]
```
Step 2: (Required) Helper script before visualization.
python attention_viz.py `--`data_dir [path_to_data_folder] `--`model_path [path_to_save_model] `--`out_dir [path_to_save_result]

Step 3: (Required) Dictionary Analysis (with correct path specified in the script).
python dictionary_analyze.py `--`dataset [SEND or SST]

### Word Cloud
For generating word cloud based on attentions (with correct path specified in the script),
```python
python word_cloud.py
```
### Highlight Sentence (Automatically Generate A Latex For You)
For generating word cloud based on attentions (with correct path specified in the script),
```python
python sentence_highlight.py
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
