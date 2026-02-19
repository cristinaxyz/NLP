from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from data import load_data, preprocess_data

def model_logistic_reg(train_ds, dev_ds, test_ds):
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

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    return y_test, y_pred