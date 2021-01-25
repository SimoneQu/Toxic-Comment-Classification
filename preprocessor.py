import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, train_path="./data/train.csv", test_path="./data/test.csv"):
        self.train_path = train_path
        self.test_path = test_path
        self.labels_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


    def get_raw_data(self):
        df_train = pd.read_csv(self.train_path)
        df_test = pd.read_csv(self.test_path)
        return df_train, df_test

    def get_features(self, df_train, df_test, method="tfidf"):
        '''
        :return
            train_features
            test_features
        '''
        ## TODO: count features
        ## TODO: tune params
        if method == "tfidf":
            tfidf = TfidfVectorizer(
                sublinear_tf=True,
                min_df=50,
                max_df=0.1,
                norm='l2',
                encoding='latin-1',
                # ngram_range=(1, 2),
                stop_words='english'
            )
            all_features = tfidf.fit_transform(
                pd.concat([df_train.comment_text, df_test.comment_text])
            ).toarray()
        else:
            raise Exception(f"method:={method} is not implemented yet")

        print("# of features:", all_features.shape[1])
        return all_features[:len(df_train)], all_features[len(df_train):]

    def get_labels(self, df_train):
        return df_train[self.labels_cols]

    def prep_data(self, method="tfidf"):
        df_train, df_test = self.get_raw_data()

        labels = self.get_labels(df_train)
        train_feat, test_feat = self.get_features(df_train, df_test, method)

        X_train, X_test, labels_train, labels_test = train_test_split(
            train_feat, labels, random_state=2021
        )
        print(X_train.shape, X_test.shape, labels_train.shape, labels_test.shape)
        return X_train, X_test, labels_train, labels_test, test_feat