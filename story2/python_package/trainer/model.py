from transformers import GPT2LMHeadModel
from trainer import metadata

def create():
    """create the model by loading a pretrained model or define your 
    own
    Args:
      num_labels: number of target labels
    """
    # Create the model, loss function, and optimizer
    model = GPT2LMHeadModel.from_pretrained(
        metadata.PRETRAINED_MODEL_NAME,
    )
    
    return model
