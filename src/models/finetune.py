#Packages
from transformers import AutoModelForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer

def distilbert_model(train_dataset, test_ds, dev_ds):
    """
    fine tuning of a pretrained model

    Args:
        train_dataset (_type_): _description_
        dev_ds (_type_): _description_
        test_ds (_type_): _description_
    """
    #GPU usage
    print(torch.cuda.is_available())
    device = 0 if torch.cuda.is_available() else -1

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(items):
        texts = [t + " " + d for t, d in zip(items["title"], items["description"])]
        return tokenizer(texts, padding="max_length", truncation=True)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)
    tokenized_dev = dev_ds.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_dev = tokenized_dev.rename_column("label", "labels")

    #Load a pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

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

    #Fine-tuning the model
    trainer.train()

    # Get predictions (logits)
    pred = trainer.predict(tokenized_test)
    logits = pred.predictions
    predicted_labels = torch.argmax(torch.tensor(logits), dim=1)

    #get true labels
    true_labels = pred.label_ids

    return true_labels, predicted_labels