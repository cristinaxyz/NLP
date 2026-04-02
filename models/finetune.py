#Packages
from transformers import AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from data import build_tokenizer


def fine_tune(train_dataset, test_ds, seed):
    """
    fine tuning of a pretrained model

    Args:
        train_dataset (_type_): _description_
        dev_ds (_type_): _description_
        test_ds (_type_): _description_
        seed (_type_): _description_
    """
    #GPU usage
    device = 0 if torch.cuda.is_available() else -1

    #Preprocess data
    model_name = "distilbert-base-uncased" #We are using distilBERT
    tokenize_fn = build_tokenizer(model_name)
    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_test = test_ds.map(tokenize_fn)
    
    #Load a pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(
        device
    )

    #Set up the trainer
    training_args = TrainingArguments(
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )