from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from module.model.naivebayes import NaiveBayes

class Trainer(object):
    def __init__(self, config, logger, classes):
        self.config = config
        self.logger = logger
        self.classes = classes
        self._model_init()

    def _model_init(self):
        if self.config['model_name'] == 'naivebayes':
            self.model = NaiveBayes(self.classes)
        else:
            self.logger.warning("Model Type:{} is not support yet".format(self.config['model_name']))

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        return self.model

    def validate(self, validate_x, validate_y):
        pred = self.model.predict(validate_x)
        accuracy = accuracy_score(validate_y, pred)
        cls_report = classification_report(validate_y, pred, zero_division=1)
        return accuracy, cls_report

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        _ = self.model.fit(train_x, train_y)
        accuracy, cls_report = self.validate(validate_x, validate_y)
        return self.model, accuracy, cls_report