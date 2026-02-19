from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from data import load_data, preprocess_data
from evaluation import evaluate_predictions

def model_linear_svm(): 
    """
    Train a linear SVM text classifier by using TF-IDF features.
    We load the AG news data, preprocesses the text, then
    convert it to TF-IDF vectrs, trains the svm on the training set.
    Then evaluates performance on the test set using accuracy, and
    macro-F1, and a confusion matrix.
    """
    train_ds, dev_ds, test_ds = load_data()

    X_train, y_train = preprocess_data(train_ds)
    X_dev, y_dev = preprocess_data(dev_ds)
    X_test, y_test = preprocess_data(test_ds)

    vectorizer = TfidfVectorizer(
        analyzer = 'word',
        ngram_range=(1, 1),
        min_df = 2,
        max_df = 0.9
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_dev_tfidf = vectorizer.transform(X_dev)
    X_test_tfidf = vectorizer.transform(X_test)

    svm = LinearSVC()
    svm.fit(X_train_tfidf, y_train)

    y_pred = svm.predict(X_test_tfidf)

    acc, macro_f1, cm = evaluate_predictions(y_test, y_pred)
    print("svm accuracy: ", acc)
    print("svm macro f1: ", macro_f1)
    print("confusion matrix: ", cm)

    return y_pred
