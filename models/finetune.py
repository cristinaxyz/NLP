#Packages
from transformers import AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from data import preprocess_data
from datasets import Dataset


def fine_tune(train_dataset, test_ds, seed):
    """
    _summary_

    Args:
        train_dataset (_type_): _description_
        dev_ds (_type_): _description_
        test_ds (_type_): _description_
        seed (_type_): _description_
    """
    #GPU usage
    device = 0 if torch.cuda.is_available() else -1

    #Preprocess data
    X_train, y_train = preprocess_data(train_dataset)
    X_test, y_test = preprocess_data(test_ds)


    #Load model
    model_name = "distilbert-base-uncased"
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

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_train,
    #     eval_dataset=tokenized_test,
    # )