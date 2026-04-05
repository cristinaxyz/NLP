from datasets import load_dataset
import re
import unicodedata
import contractions
from collections import Counter

import torch 
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass

@dataclass
class Batch:
    x: torch.Tensor
    lengths: torch.Tensor
    y: torch.Tensor

def load_data(seed):
    ds = load_dataset("sh0416/ag_news")
    split = ds['train'].train_test_split(test_size = 0.1, seed = seed)
    training_ds = split['train']
    dev_ds = split['test']
    test_ds = ds['test']
    return training_ds, dev_ds, test_ds

def normalization(text):
    text = text.lower()

    # Step 1: remove all the special characters from the text -- remaining, words, letters, spaces
    text = re.sub(r"[^\w\s]", "", text)

    #Step 2: remove accents
    nfkd_form = unicodedata.normalize("NFKD", text)
    text = "".join([char for char in nfkd_form if not unicodedata.combining(char)])

    #Step 3: Expanding contractions
    text = contractions.fix(text)

    #Step 4: Tokenize
    tokenized = tokenize(text)

    return " ".join(tokenized)

def tokenize(text):
    """
    Meaningful units: words.
    """
    text = text.split()
    return text

def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(examples):
        return tokenizer(examples["text"], padding=True, truncation=True)

    return tokenize_batch

def preprocess_data(dataset):
    """
    For each dataset consider x and y (labels).
    """
    X = []
    y = []

    for element in dataset:
        title = element["title"]
        description = element["description"]
        text = normalization(title) + " " + normalization(description)
        X.append(text)
        y.append(int(element["label"])) 

    return X, y

PAD = "<pad>"
UNK = "<unk>"

def build_vocab(texts, min_freq: int = 2, max_size: int = 30000) -> dict:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
 
    vocab = {PAD: 0, UNK: 1}

    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab

def numericalize(tokens, vocab): 
    return [vocab.get(token, vocab[UNK]) for token in tokens]

class TextDataset(Dataset): 
    def __init__(self, dataset, vocab, max_length=200): 
        self.dataset = dataset
        self.vocab = vocab 
        self.max_length = max_length
    
    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, idx): 
        item = self.dataset[idx]

        text = normalization(item["title"]) + " " + normalization(item["description"])
        tokens = tokenize(text)

        if len(tokens) == 0: 
            ids = [self.vocab[UNK]]
        else: 
            ids = numericalize(tokens, self.vocab)[: self.max_length]
            if len(ids) == 0: 
                ids = [self.vocab[UNK]]
        
        label = int(item["label"]) 
        return ids, label 
    
def collate_fn(batch, pad_idx):
    lengths = torch.tensor([len(ids) for ids, _ in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    x = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    y = torch.tensor([label for _, label in batch], dtype=torch.long)

    for i, (ids, _) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    return Batch(x=x, lengths=lengths, y=y)

def build_loaders(train_ds, dev_ds, test_ds, batch_size=32, max_length=200):
    trained_texts, _ = preprocess_data(train_ds)
    vocab = build_vocab(trained_texts, min_freq=2, max_size=30000)
    #vocab_size = len(vocab)
    pad_idx = vocab[PAD]

    train_ds = TextDataset(train_ds, vocab, max_length=max_length)
    dev_ds = TextDataset(dev_ds, vocab, max_length=max_length)
    test_ds = TextDataset(test_ds, vocab, max_length=max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )

    return train_loader, dev_loader, test_loader, vocab, pad_idx


