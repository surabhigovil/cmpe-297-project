
import os
import json
import logging
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
from ts.torch_handler.base_handler import BaseHandler
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logger = logging.getLogger(__name__)


class MyHandler(BaseHandler):
    """
    The handler takes an input string and returns the classification text 
    based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self,ctx):
        """ Loads the model.pt file and initialized the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """    
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt or pytorch_model.bin file")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        print('loading model and tokenizer')
        # Ensure to use the same tokenizer used during training

        self.initialized = True

    def preprocess(self, data):
        """ Preprocessing input request by tokenizing
            Extend with your own preprocessing steps as needed
        """
        print("preprocess")
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        print("text is",text)
        s = text.decode('utf-8')
        logger.info("Received text: '%s'", s)
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
        
        t=list(s)
        logger.info("sentence: '%s'",s)
        tokenizer_args = ((s,))
        logger.info("tokenizer args: %s", *tokenizer_args)
        encoded_prompt = tokenizer.encode(*tokenizer_args, add_special_tokens=False, return_tensors="pt")
        print("preprocess done")
        print(encoded_prompt)
        return encoded_prompt

    def inference(self, inputs):
        """ Predict the class of a text using a trained transformer model.
        """
        print('inference')
        print(inputs)
        output_sequences = self.model.generate(
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
        print('inference end')
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()
            # Decode text
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token=tokenizer.eos_token
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            # Remove all text after eos token
            text = text[: text.find(tokenizer.eos_token)]
            texts.append(text)
            print(text)
        
        return [texts]
        

    def postprocess(self, inference_output):
        print('post')
        return inference_output
    # def handle(self, data, ctx):
    #     model_input = self.preprocess(data)
    #     model_output = self.inference(model_input)
    #     return self.postprocess(model_output)
        
