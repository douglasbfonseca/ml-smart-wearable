import argparse
import logging
import logging.config

import yaml

from src.initial_transformer import data_transformer
from src.pipeline import MLReport

def main():
    """
    Entry point to run the pipeline
    """
    
    # Parsing YAML file
    parser = argparse.ArgumentParser(description='Run Pipeline')
    parser.add_argument('config', help='A configuration file in YAML format.')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    # Configure logging
    log_config = config['logging']
    logging.config.dictConfig(log_config)
    logger = logging.getLogger(__name__)

    # Configure data transformer
    initial_transformer = data_transformer

    # Data path
    path = config['path']

    logger.info('Job started')
    ml_report = MLReport(initial_transformer,
                         path)
    ml_report.run_pipeline()
    

if __name__ == '__main__':
    main()