from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from data import load_data, preprocess_data
from evaluation import evaluate_predictions

def model_logistic_reg():
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

    X_train_tfidf = vectorizer.transform(X_train)
    X_dev_tfidf = vectorizer.transform(X_dev)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_train_tfidf)

    return y_pred

#   we may compute directly here the evaluation metrics 
#   acc, macro_f1, cm = evaluate_predictions(y_test, y_pred)
#   print("Accuracy:", acc)
#   print("Macro F1:", macro_f1)
#   print("Confusion Matrix:", cm)