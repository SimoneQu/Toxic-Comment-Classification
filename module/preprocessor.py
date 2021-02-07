import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Preprocessor:
    def __init__(self, config, logger):
        self.config = config['preprocessing']
        self.nn_params = config.get('nn_params', None)
        self.logger = logger
        self.classes = self.config['classes']
        self._load_raw_data()
        self.vocab_size = None
        self.embedding_matrix = None

    def _load_raw_data(self):
        self.df_train = pd.read_csv(self.config['dir_traindata'])
        self.df_test = pd.read_csv(self.config['dir_testdata'])
        self.test_ids = self.df_test['id']
        return

    def prep_data(self):
        prep_method = self.config.get('prep_method', None)

        if prep_method == None:
            data_x = self.df_train.comment_text.to_numpy()
            test_x = self.df_test.comment_text.to_numpy()
        elif prep_method == "tfidf_vectorization":
            print("processing tfidf_vectorization...")
            data_x, test_x = self.tfidf_vectorization()
        elif prep_method == "count_vectorization":
            print("processing count_vectorization...")
            data_x, test_x = self.count_vectorization()
        elif prep_method == "nn_vectorization":
            print("processing nn_vectorization...")
            data_x, test_x = self.nn_vectorization()
        else:
            raise Exception(f"method:={prep_method} is not implemented yet")

        data_y = self.df_train[self.classes]

        train_x, validate_x, train_y, validate_y = train_test_split(
            data_x, data_y, random_state=self.config['random_seed']
        )

        return data_x, data_y, train_x, train_y, validate_x, validate_y, test_x

    def tfidf_vectorization(self):
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

    def count_vectorization(self):
        vectorizer = CountVectorizer()
        all_features = vectorizer.fit_transform(
                pd.concat([self.df_train.comment_text, self.df_test.comment_text])
        ).toarray()
        return all_features[:len(self.df_train)], all_features[len(self.df_train):]

    def nn_vectorization(self):
        params = self.nn_params
        all_text = pd.concat([self.df_train.comment_text, self.df_test.comment_text])
        tokenizer = Tokenizer(num_words=params['token_num_words'])
        tokenizer.fit_on_texts(all_text)
        self.vocab_size = min(params['token_num_words'], len(tokenizer.word_index)) + 1

        all_features = tokenizer.texts_to_sequences(all_text)
        all_features = pad_sequences(all_features, padding='post', maxlen=params['sentence_maxlen'])

        pretrained_embedding_name = params.get("pretrained_embedding", None)
        if pretrained_embedding_name:
            print("load pretrained embedding {}...".format(pretrained_embedding_name))
            import gensim.downloader as api
            wv = api.load(pretrained_embedding_name)

            print("creating embedding matrix...")
            self.embedding_matrix = np.zeros((self.vocab_size, params['embedding_dim']))
            for word, i in tokenizer.word_index.items():
                if i >= self.vocab_size:
                    continue
                if word in wv.vocab:
                    self.embedding_matrix[i] = wv.word_vec(word)

        return all_features[:len(self.df_train)], all_features[len(self.df_train):]

