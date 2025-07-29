## Overview

This repository contains code and resources for a Ted Talks Classifier.
In addition to classifying Ted talks we also did the following:
1) Tested performance on titles only (New model)
2) Tested on news articles and podcasts
3) Tested effect on transcript length
4) Tested performance on different percentages of transcript



## Repository Structure

* **clean\_transcript.py**: Script for preprocessing and cleaning transcript data.
* **compute\_metrics.py**: Functions to compute evaluation metrics (precision, recall, F1-score) for model predictions.
* **constants.py**: Contains global constants used throughout the project.
* **get\_files.py**: Utility functions to retrieve and manage dataset files.
* **load\_data\_to\_pickle.py**: Script to load and preprocess data, storing it efficiently in pickle format.
* **split\_train\_test.py**: Functionality to split datasets into training and test sets.
* **ted\_dataset.py**: Custom dataset class for loading and handling the TED dataset for NLP tasks.
* **train\_models.py**: Training script
* **test\_models.py**: Testing script

  
Install dependencies:

```bash
pip install -r requirements.txt
```



### Training

Train the Transformer model using the following command:

```bash
python code/train_models.py --is_full_transcript_model <bool> --save_model_name <string> 
```
In order to train the transcript model put in is_full_transcript_model as true, save_model_name determines where you'll save it.
Important to note that this training requires the a A-100 GPU and can take an enourmous amount of time


### Evaluation

Evaluate the model performance with:

```bash
python code/test_models.py --from_scratch <bool> --is_full_transcript_model <bool> --manual_model <bool> --model_name <string|None>
```

Here we evaluate our model. We can either test the transcript model or the title model depending on what we set is_full_transcript_model

For transcript model we are testing:
1) Global Metrics
2) Label Metrics
3) Reliability
4) Performance for duration
5) Metrics by percent
6) Performance of news articles and podcasts (both have been put into datasets which fit our model)

While for the title model we are exclusively testing the global metrics.

If we want to test our own model we set manual_model to True and give in a model_name (Default for manual_model is False and model_name is "")

If we want to test using the given outputs our group has obtained set from_scratch to False. To test from scratch set from_scratch as True. 
It is important to note that testing from scratch can potentially take hours and many computers can't handle the memory alloacation needed.
