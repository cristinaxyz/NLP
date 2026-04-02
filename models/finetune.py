#Packages
from transformers import AutoModelForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from data import build_tokenizer


def fine_tune(train_dataset, test_ds, dev_ds):
    """
    fine tuning of a pretrained model

    Args:
        train_dataset (_type_): _description_
        dev_ds (_type_): _description_
        test_ds (_type_): _description_
    """
    #GPU usage
    device = 0 if torch.cuda.is_available() else -1

    #Preprocess data
    model_name = "distilbert-base-uncased" #We are using distilBERT
    tokenize_fn = build_tokenizer(model_name)
    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_test = test_ds.map(tokenize_fn)
    
    #Load a pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(
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

    #Fine-tuning the model
    trainer.train()

    # Get predictions (logits)
    with torch.no_grad():  # Disable gradient computation since we're just doing inference
        outputs = model(**dev_ds) #I'm not sure which ds to use, so Im just leaving this for tomorrow
        logits = outputs.logits #also, this is just for one item, so we have to edit that as well

    predicted_label = torch.argmax(logits, dim=1).item()


    return predicted_label