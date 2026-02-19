from datasets import load_dataset
import re
import unicodedata
import contractions
from nltk.stem import WordNetLemmatizer

import nltk

# Ensure WordNet and omw-1.4 are downloaded
try:
    from nltk.corpus import wordnet
    _ = wordnet.synsets('test')  # test if WordNet is available
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

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

    # Step 1: remove all the special characters from the text -- remaining, words, letters, spaces
    no_syms = re.sub(r"[^\w\s]", "", text)

    #Step 2: remove accents
    nfkd_form = unicodedata.normalize("NFKD", no_syms)
    no_acc = "".join([char for char in nfkd_form if not unicodedata.combining(char)])

    #Step 3: Expanding contractions
    expanded = contractions.fix(no_acc)

    #Step 4: Tokenize
    tokenized = tokenize(expanded)

    #Step 5: Lemmatize
    lemmatizer = WordNetLemmatizer()
    outp = "".join(lemmatizer.lemmatize(token) for token in tokenized)
    return outp

def tokenize(text):
    """
    Meaningful units: words.
    """
    text = text.split()
    return text

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
        y.append(element["label"])

    return X, y