from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from module.model.naivebayes import NaiveBayes
from module.model.rnn import RNN

class Trainer(object):
    def __init__(self, config, logger, classes, vocab_size, embedding_matrix):
        self.config = config
        self.logger = logger
        self.classes = classes
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self._model_init()

    def _model_init(self):
        if self.config['model_name'] == 'naivebayes':
            self.model = NaiveBayes(self.classes)
        elif self.config['model_name'] == 'rnn':
            self.model = RNN(self.classes, self.vocab_size, self.embedding_matrix, self.config['nn_params'], self.logger)
        else:
            self.logger.warning("Model Type:{} is not support yet".format(self.config['model_name']))

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        return self.model

    def metrics(self, predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        cls_report = classification_report(labels, predictions, zero_division=1)
        return accuracy, cls_report

    def validate(self, validate_x, validate_y):
        pred = self.model.predict(validate_x)
        return self.metrics(pred, validate_y)

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        pred, history = self.model.fit_and_validate(train_x, train_y, validate_x, validate_y)
        accuracy, cls_report = self.metrics(pred, validate_y)
        return self.model, accuracy, cls_report, history