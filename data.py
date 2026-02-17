from datasets import load_dataset

def load_data():
    ds = load_dataset("sh0416/ag_news")
    seed = 42
    split = ds['train'].train_test_split(test_size = 0.1, seed = seed)
    training_ds = split['train']
    dev_ds = split['test']
    test_ds = ds['test']
    return training_ds, dev_ds, test_ds

def normalization(text):
    text = text.lower()
    return text

def tokenize(text):
    """
    Meaningful units: words.
    """
    text = text.split()
    return text

def preprocess_data():
    """
    For each dataset consider x and y (labels).
    """
    pass