from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from data import preprocess_data
from sklearn.metrics import accuracy_score

def model_linear_svm(train_ds, dev_ds, test_ds, seed): 
    """
    Train a linear SVM text classifier by using TF-IDF features.
    We load the AG news data, preprocesses the text, then
    convert it to TF-IDF vectrs, trains the svm on the training set.
    Then evaluates performance on the test set using accuracy, and
    macro-F1, and a confusion matrix.
    """

#preprocess 
    X_train, y_train = preprocess_data(train_ds)
    X_dev, y_dev = preprocess_data(dev_ds)
    X_test, y_test = preprocess_data(test_ds)

    best_parameters = None
    best_acc = 0 

    ngram_options = [(1, 1), (2, 2)]
    min_df_options = [1, 2, 5]
    max_df_options = [0.8, 0.9]

    for ngram in ngram_options:
        for min_df in min_df_options:
            for max_df in max_df_options:

                vectorizer = TfidfVectorizer(
                    ngram_range=ngram, 
                    min_df = min_df,
                    max_df = max_df)
                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_dev_tfidf = vectorizer.transform(X_dev)
                model = LinearSVC(max_iter = 1000, random_state=seed)
                model.fit(X_train_tfidf, y_train)
                y_dev_pred = model.predict(X_dev_tfidf)
                dev_acc = accuracy_score(y_dev_pred, y_dev)
                if dev_acc> best_acc:
                    best_acc = dev_acc
                    best_parameters = (ngram, min_df, max_df)
    
    print("Best parameters for TD-IDF + Linear SVM:", best_parameters)

    best_vectorizer = TfidfVectorizer(
                    analyzer = 'word',
                    ngram_range=best_parameters[0], 
                    min_df = best_parameters[1],
                    max_df = best_parameters[2])
    
    X_train_tfidf = best_vectorizer.fit_transform(X_train)
    X_test_tfidf = best_vectorizer.transform(X_test)

#model
    svm = LinearSVC()
    svm.fit(X_train_tfidf, y_train)

#prediction
    y_pred = svm.predict(X_test_tfidf)

    return y_test, y_pred