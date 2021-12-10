import os
import datetime

from google.cloud import storage
from transformers import GPT2Tokenizer
from datasets import load_dataset, load_metric, ReadInstruction
from trainer import metadata, model
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch
class StoryDataset:
    def __init__(self, inputs):
        self.ids = inputs['input_ids'][:12]
        self.attention_mask = inputs['attention_mask'][:12]
        self.labels=inputs['labels'][:12]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):

        return [torch.tensor(self.ids[item], dtype=torch.long),
                torch.tensor(self.attention_mask[item], dtype=torch.long),
                torch.tensor(self.labels[item], dtype=torch.long)]


def getLines(filename):
    client = storage.Client()
    bucket_name = 'story-text1'
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(filename)
    lines=[]
    downloaded_blob = blob.download_as_string()
    downloaded_blob = downloaded_blob.decode('utf-8')
    for l in downloaded_blob.split('\n'):
        lines.append(l[7:])
    return lines
    
def combinetext(prompt, story):
    prompts=getLines(prompt)
    stories=getLines(story)
    assert len(prompts)==len(stories)
    combine=[]
    for i in range(len(prompts)):
        combine.append(prompts[i].rstrip()+' <sep> '+" ".join(stories[i].split()[:300]))
    return combine

#do a littel text clean with punctuations
def cleanpunctuation(s):
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
    return s
def preprocess_function(examples):

    tokenizer = GPT2Tokenizer.from_pretrained(metadata.PRETRAINED_MODEL_NAME)
    tokenizer.pad_token=tokenizer.eos_token
        # Tokenize the texts
    result = tokenizer(examples, padding=True,truncation=True,max_length=metadata.MAX_SEQ_LENGTH)

    return result

def create_labels(inputs):
    labels=[]
    for ids,attention_mask in zip(inputs['input_ids'],inputs['attention_mask']):
        label=ids.copy()
        real_len=sum(attention_mask)
        padding_len=len(attention_mask)-sum(attention_mask)
        label[:]=label[:real_len]+[-100]*padding_len
        labels.append(label)
    inputs['labels']=labels
    return inputs

def load_data(args):
    """Loads the data into two different data loaders. (Train, Test)
        Args:
            args: arguments passed to the python script
    """
    # Dataset loading repeated here to make this cell idempotent
    # Since we are over-writing datasets variable

    train_text=combinetext('valid.wp_source', 'valid.wp_target')
    train_text=list(map(cleanpunctuation,train_text))
    valid_text=combinetext('test.wp_source', 'test.wp_target')
    valid_text=list(map(cleanpunctuation,valid_text))
    train_text=preprocess_function(train_text)
    valid_text=preprocess_function(valid_text)
    inputs_train=create_labels(train_text)
    inputs_valid=create_labels(valid_text)
    traindata=StoryDataset(inputs_train)
    validdata=StoryDataset(inputs_valid)
    train_dataloader = torch.utils.data.DataLoader(
    traindata,
    shuffle=False,
    batch_size=args.batch_size)
    num_training_steps_per_epoch=len(train_dataloader)
    # print(num_training_steps_per_epoch)

    return traindata, validdata,num_training_steps_per_epoch

def get_optimizers(args,num_training_steps_per_epoch):
    m=model.create()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in m.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in m.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    ]
    # print(num_training_steps_per_epoch)
    total_num_training_steps = int(num_training_steps_per_epoch*args.num_epochs)
    warmup_steps=int(total_num_training_steps*args.warmup)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_num_training_steps
    )
    return optimizer,scheduler

def save_model(args):
    """Saves the model to Google Cloud Storage or local file system
    Args:
      args: contains name for saved model.
    """
    scheme = 'gs://'
    if args.job_dir.startswith(scheme):
        job_dir = args.job_dir.split("/")
        bucket_name = job_dir[2]
        object_prefix = "/".join(job_dir[3:]).rstrip("/")

        if object_prefix:
            model_path = '{}/{}'.format(object_prefix, args.model_name)
        else:
            model_path = '{}'.format(args.model_name)

        bucket = storage.Client().bucket(bucket_name)    
        local_path = os.path.join("/tmp", args.model_name)
        files = [f for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))]
        for file in files:
            local_file = os.path.join(local_path, file)
            blob = bucket.blob("/".join([model_path, file]))
            blob.upload_from_filename(local_file)
        print(f"Saved model files in gs://{bucket_name}/{model_path}")
    else:
        print(f"Saved model files at {os.path.join('/tmp', args.model_name)}")
        print(f"To save model files in GCS bucket, please specify job_dir starting with gs://")


