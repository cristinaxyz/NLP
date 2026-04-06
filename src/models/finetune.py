#Packages
from transformers import AutoModelForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

def distilbert_model(train_ds, dev_ds, test_ds, learning_rate=2e-5, batch_size=8):
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
        output_dir="trainer_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=5,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")

        return {
            "accuracy": acc,
            "f1": f1,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    #Fine-tuning the model
    #trainer.train()

    eval_results = trainer.evaluate(tokenized_dev)
    print(eval_results)

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


from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

def distilbert_predict_from_checkpoint(dev_ds, test_ds, checkpoint_path):
    def shift_labels(example):
        example["label"] = example["label"] - 1
        return example

    dev_ds = dev_ds.map(shift_labels)
    test_ds = test_ds.map(shift_labels)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(items):
        texts = [t + " " + d for t, d in zip(items["title"], items["description"])]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized_dev = dev_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)

    tokenized_dev = tokenized_dev.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    training_args = TrainingArguments(
        output_dir="trainer_output_eval",
        report_to="none",
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    dev_pred = trainer.predict(tokenized_dev)
    dev_predicted_labels = torch.argmax(torch.tensor(dev_pred.predictions), dim=1)
    dev_true_labels = dev_pred.label_ids

    test_pred = trainer.predict(tokenized_test)
    test_predicted_labels = torch.argmax(torch.tensor(test_pred.predictions), dim=1)
    test_true_labels = test_pred.label_ids

    return dev_true_labels, dev_predicted_labels, test_true_labels, test_predicted_labels