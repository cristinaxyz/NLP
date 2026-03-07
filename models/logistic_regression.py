from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from data import preprocess_data

def model_logistic_reg(train_ds, dev_ds, test_ds, seed):
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
                    analyzer = 'word',
                    ngram_range=ngram, 
                    min_df = min_df,
                    max_df = max_df)
                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_dev_tfidf = vectorizer.transform(X_dev)
                model = LogisticRegression(max_iter = 1000)
                model.fit(X_train_tfidf, y_train)
                y_dev_pred = model.predict(X_dev_tfidf)
                dev_acc = accuracy_score(y_dev_pred, y_dev)
                if dev_acc> best_acc:
                    best_acc = dev_acc
                    best_parameters = (ngram, min_df, max_df)
    
    print("Best parameters for TD-IDF + Logistic Regression:", best_parameters)

    best_vectorizer = TfidfVectorizer(
                    ngram_range=best_parameters[0], 
                    min_df = best_parameters[1],
                    max_df = best_parameters[2])
    
    X_train_tfidf = best_vectorizer.fit_transform(X_train)
    X_test_tfidf = best_vectorizer.transform(X_test)
    
    final_model = LogisticRegression(max_iter = 1000, random_state=seed)
    final_model.fit(X_train_tfidf, y_train)

    y_pred = final_model.predict(X_test_tfidf)

    return y_test, y_pred