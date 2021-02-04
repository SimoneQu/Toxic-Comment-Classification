import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.classes = self.config['classes']
        self._load_raw_data()

    def _load_raw_data(self):
        self.df_train = pd.read_csv(self.config['dir_traindata'])
        self.df_test = pd.read_csv(self.config['dir_testdata'])
        self.test_ids = self.df_test['id']
        return

    def prep_data(self):
        prep_method = self.config.get('prep_method', None)

        if prep_method == "tfidf":
            print("processing: generating tfidf features")
            data_x, test_x = self.get_tfidf_features()
            data_y = self.df_train[self.classes]
        else:
            raise Exception(f"method:={prep_method} is not implemented yet")

        train_x, validate_x, train_y, validate_y = train_test_split(
            data_x, data_y, random_state=self.config['random_seed']
        )

        return data_x, data_y, train_x, train_y, validate_x, validate_y, test_x

    def get_tfidf_features(self):
        params = self.config['tfidf_params']
        tfidf = TfidfVectorizer(
            sublinear_tf=True,
            min_df=params['min_df'],
            max_df=params['max_df'],
            norm='l2',
            encoding='latin-1',
            stop_words='english'
        )
        all_features = tfidf.fit_transform(
            pd.concat([self.df_train.comment_text, self.df_test.comment_text])
        ).toarray()

        print("# of features:", all_features.shape[1])
        return all_features[:len(self.df_train)], all_features[len(self.df_train):]


