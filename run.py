import argparse
import logging
import logging.config

import yaml

from src.initial_transformer import DataTransformer
from src.save_results import SaveResults
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

    # Data config transformer parameters
    path = config['path']
    overlap = config['overlap']
    window_param = config['window_param']

    # Configure data transformer class
    initial_transformer = DataTransformer(path,
                                          overlap,
                                          window_param)

    # Configure save results class
    results = SaveResults()

    #Running the job
    logger.info('Job started')
    ml_report = MLReport(initial_transformer,
                         results)
    ml_report.run_pipeline()
    logger.info('Job finished')
    

if __name__ == '__main__':
    main()