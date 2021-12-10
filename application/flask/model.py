
import os
import json
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np

from deep_daze import Imagine





def initialize():
    model_dir='./model_artifacts' 
    model_pt_path = os.path.join(model_dir, 'pytorch_model.bin')
    if not os.path.isfile(model_pt_path):
        raise RuntimeError("Missing the model.pt or pytorch_model.bin file")
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return model

def preprocess(data):
    s=data
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token=tokenizer.eos_token
        # Tokenize the texts
    for p in '!,.:;?':
        s=s.replace(' '+p,p)
        s=s.replace(' '+'n\'t','n\'t')
        s=s.replace(' '+'\'s','\'s')
        s=s.replace(' '+'\'re','\'re')
        s=s.replace(' '+'\'ve','\'ve')
        s=s.replace(' '+'\'ll','\'ll')
        s=s.replace(' '+'\'am','\'am')
        s=s.replace(' '+'\'m','\'m')
        s=s.replace(' '+'\' m','\'m')
        s=s.replace(' '+'\'m','\'m')
        s=s.replace(' '+'\' ve','\'ve')
        s=s.replace(' '+'\' s','\'s')
        s=s.replace('<newline>','\n')
        
    s=((s,))
    encoded_prompt = tokenizer.encode(*s, add_special_tokens=False, return_tensors="pt")
    return encoded_prompt

def inference(inputs,model):
    output_sequences = model.generate(
        input_ids=inputs,
        max_length=300,
        temperature=1,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=3
    )
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()
    texts=[]
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()
            # Decode text
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token=tokenizer.eos_token
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            # Remove all text after eos token
        text = text[: text.find(tokenizer.eos_token)]
        texts.append(text)
        # print(text)
        
    return [texts]
        
def postprocess(inference_output):
    print('post')
    return inference_output

def generate_images(text):
  imagine = Imagine(
    text = text,
    num_layers = 24,
    create_story=True
    )
  print(imagine)
def handle(data):
    print('handle')
    model=initialize()
    model_input = preprocess(data)
    model_output = inference(model_input,model)
    #generate_images(model_output[0][0][0])
    return postprocess(model_output)


        


# handle('Childrens logic dictates the way the world works.')
