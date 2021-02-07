import yaml
import logging
# import argparse
from module.preprocessor import Preprocessor
from module.trainer import Trainer
from module.utils import *


# from module import Preprocessor
# import importlib
# importlib.reload(module.trainer)
# importlib.reload(Trainer)

if __name__ == "__main__":

    # load config
    with open('./config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    logger = init_logger(config)

    # preprocessing
    pp = Preprocessor(config, logger)
    data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = pp.prep_data()

    # training
    trainer = Trainer(config, logger, pp.classes, pp.vocab_size, pp.embedding_matrix)
    model, accuracy, cls_report, history = trainer.fit_and_validate(train_x, train_y, validate_x, validate_y)
    if history:
        logger.info(history.history)
    logger.info("accuracy:{}".format(accuracy))
    logger.info("\n{}\n".format(cls_report))

    # # output
    # model = trainer.fit(data_x, data_y)
    # filename = get_output_filename(config)
    # output_predictions(model, pp.test_ids, test_x, filename)
