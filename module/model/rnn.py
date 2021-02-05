import tensorflow as tf
import pandas as pd

class RNN(object):
    def __init__(self, classes, params):
        self.params = params
        self.classes = classes
        self.model = self._build()

    def _build(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=self.params['token_num_words'],
                output_dim=self.params['embedding_dim'],
                input_length=self.params['sentence_maxlen'],
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.classes), activation='sigmoid')
        ])
        model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
        )
        model.summary()
        return model

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train, epochs=self.params['epochs'])

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        history = self.model.fit(
            train_x, train_y,
            epochs=self.params['epochs'],
            verbose=True,
            validation_data=(validate_x, validate_y),
        )
        predictions = self.predict(validate_x)
        return predictions, history

    def predict(self, X_test):
        pred_probs = self.model.predict(X_test)
        pred = pred_probs > 0.5
        df_pred = pd.DataFrame(pred, columns=self.classes)
        return df_pred

    def predict_prob(self, X_test):
        pred_probs = self.model.predict(X_test)
        df_pred_probs = pd.DataFrame(pred_probs, columns=self.classes)
        return df_pred_probs