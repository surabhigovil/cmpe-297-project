import os
import numpy as np
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel
)
import transformers
from transformers import TrainingArguments, Trainer
# from transformers import Trainer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from trainer import model, metadata, utils

def dummy_data_collector(features):
    batch = {}
    batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['attention_mask'] = torch.stack([f[1] for f in features])
    batch['labels'] = torch.stack([f[2] for f in features])
    batch['input_ids']=batch['input_ids'][:12]
    batch['attention_mask']=batch['attention_mask'][:12]
    batch['labels']=batch['labels'][:12]  
    
    return batch

def train(args, model, traindata, validdata,optimizer,scheduler):

    training_args = TrainingArguments(
    output_dir=os.path.join("/tmp", args.model_name),
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    evaluation_strategy="epoch",
    no_cuda=True
    )

    opti=optimizer,scheduler

    # print(opti)
    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=dummy_data_collector,
    train_dataset=traindata,
    eval_dataset=validdata,
    optimizers=opti
    )
    trainer.train()

    return trainer
    
    


def run(args):
    """Load the data, train, evaluate, and export the model for serving and
     evaluating.
    Args:
      args: experiment parameters.
    """
    # Open our dataset
    print('loading data')
    train_dataset, test_dataset,num_training_steps_per_epoch = utils.load_data(args)
    print('loading data completed')
    optimizer,scheduler=utils.get_optimizers(args,num_training_steps_per_epoch)
    
    
    # Create the model, loss function, and optimizer
    text_generator= model.create()
    
    # Train / Test the model
    trainer = train(args, text_generator, train_dataset, test_dataset,optimizer,scheduler)

    # Export the trained model
    trainer.save_model(os.path.join("/tmp", args.model_name))

    # Save the model to GCS
    if args.job_dir:
        utils.save_model(args)
    else:
        print(f"Saved model files at {os.path.join('/tmp', args.model_name)}")
        print(f"To save model files in GCS bucket, please specify job_dir starting with gs://")