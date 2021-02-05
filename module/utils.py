import pandas as pd
import logging


def output_predictions(model, test_id, test_x, out_dir):
    pred = model.predict_prob(test_x)
    df_out = pd.concat([test_id.to_frame(),pred],axis=1)
    df_out.to_csv(out_dir,index=False)


def get_output_filename(config):
    out_file = '{}{}_{}.csv'.format(
        config['output_dir'], config['model_name'], config['model_version']
    )
    return out_file


def init_logger(config):
    if config['preprocessing']['dir_traindata'] == './data/train.csv':
        logger_file = 'model_history.log'
    else:
        logger_file = 'temp_model_history.log'

    logging.basicConfig(filename=logger_file, format='%(asctime)-15s %(message)s', level="INFO")
    logger = logging.getLogger('global_logger')

    logger.info("-"*50)
    logger.info("model info for {}.{}".format(config['model_name'], config['model_version']))
    logger.info("")

    logger.info("config info:")
    for key, val in config.items():
        logger.info(key)
        logger.info(val)

    logger.info("")
    return logger