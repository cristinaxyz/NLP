import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from data import preprocess_data


def cnn_model(train_ds, dev_ds, test_ds, seed):

    #Step 1: preprocess
    X_train, y_train = preprocess_data(train_ds)
    X_dev, y_dev = preprocess_data(dev_ds)
    X_test, y_test = preprocess_data(test_ds)

    #Step 2: set parameters
    vocab_size = 10000 #max numbers of words in vocab
    max_length = 500 #max sequence of length
    embedding_dim = 100 #word vector dimension
    filters = 128 #the amount of CNN filters
    kernel_size = 5
    dense_units = 64 #number of neurons in dense layer
    dropout_rate = 0.5
    batch_size = 32
    epochs = 10


    model = Sequential([
        #convert words into vectors
        Embedding(vocab_size, embedding_dim, input_length=max_length), 
         #detects local patterns in word sequences
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        #reduces the output to its strongest signal, prevents overfitting 
        GlobalMaxPooling1D(), 
        #combines all features from before and finds their relationship
        Dense(dense_units, activation='relu'),
         #regularization to prevent overfitting 
        Dropout(dropout_rate),
         #probability for binary classification
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    #Step 4: train model
    model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True
    )

    #Step 5: predict on test set
    y_pred_prob = model.predict(X_test)
    # converts probabilities to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    return y_test, y_pred