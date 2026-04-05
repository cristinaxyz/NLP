#Packages
from transformers import AutoModelForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer

def distilbert_model(train_ds, dev_ds, test_ds, learning_rate=2e-5):
    """
    fine tuning of a pretrained model

    Args:
        train_dataset (_type_): _description_
        dev_ds (_type_): _description_
        test_ds (_type_): _description_
    """
    def shift_labels(example):
        example["label"] = example["label"] - 1
        return example

    train_ds = train_ds.map(shift_labels)
    dev_ds   = dev_ds.map(shift_labels)
    test_ds  = test_ds.map(shift_labels)
    print(sorted(set(train_ds["label"])))

    #GPU usage
    print(torch.cuda.is_available())
    device = 0 if torch.cuda.is_available() else -1

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(items):
        texts = [t + " " + d for t, d in zip(items["title"], items["description"])]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)
    tokenized_dev = dev_ds.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_dev = tokenized_dev.rename_column("label", "labels")

    #Load a pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    #Set up the trainer
    training_args = TrainingArguments(
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
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

    #Fine-tuning the model
    trainer.train()

    #this is for the parameter selection
    dev_pred = trainer.predict(tokenized_dev)
    dev_logits = dev_pred.predictions
    dev_predicted_labels = torch.argmax(torch.tensor(dev_logits), dim=1)
    dev_true_labels = dev_pred.label_ids

    # Get final redictions (logits)
    pred = trainer.predict(tokenized_test)
    logits = pred.predictions
    predicted_labels = torch.argmax(torch.tensor(logits), dim=1)
    true_labels = pred.label_ids

    return dev_true_labels, dev_predicted_labels, true_labels, predicted_labels


