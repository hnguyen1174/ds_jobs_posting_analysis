from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, pipeline
import pandas as pd
import numpy as np
import torch
import sys

if __name__ == '__main__':

    skill_model_dict = {
        'Computer Vision': 'cv_distilbert',
        'Machine Learning': 'ml_distilbert',
        'Natural Language Processing': 'nlp_distilbert'
    }

    # Sample: text = 'This job requires machine learning and NLP!'
    text = str(sys.argv[0])

    for skill, model in skill_model_dict.items():

        model_name = '../models/{}'.format(model)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        tokenizer = AutoTokenizer.from_pretrained(model_name + '/tokenizer')

        input_tensor = tokenizer.encode(text, return_tensors="pt")
        logits = model(input_tensor)[0][0][0]
        score = float(logits.cpu().detach().numpy())
        
        print('{} Score: {}/300.'.format(skill, round(score, 2)))