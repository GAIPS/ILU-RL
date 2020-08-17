import sys
import argparse
from os import environ
import multiprocessing as mp
import configparser

from pathlib import Path

from ilurl.loaders.xml2csv import main as _xml2csv

ILURL_HOME = environ['ILURL_HOME']
CONFIG_PATH = \
    Path(f'{ILURL_HOME}/config/')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str, nargs='?',
                        help='''A directory which it\'s subdirectories are train runs.''')

    parsed = parser.parse_args()
    sys.argv = [sys.argv[0]]

    return parsed


def convert(file_path):
    csv_path = str(file_path).replace('xml', 'csv')
    args = [str(file_path), '-o', csv_path]
    try:
        _xml2csv(args)
        Path(file_path).unlink()
    except Exception:
        raise

def xml2csv(experiment_root_path=None):

    if not experiment_root_path:
        # Parse command line.
        args = get_arguments()
        experiment_root_path = args.experiment_dir
        num_processors = mp.cpu_count()
    else:
        run_config = configparser.ConfigParser()
        run_config.read(str(CONFIG_PATH / 'run.config'))
        num_processors = int(run_config.get('run_args', 'num_processors'))

    print('\nConverting .xml files to .csv ...\n')

    # Get all .xml files recursively.
    xml_paths = list(Path(experiment_root_path).rglob('*.xml'))
    xml_paths = [str(p) for p in xml_paths]

    # Convert files.
    if num_processors > 1:
        pool = mp.Pool(num_processors)
        pool.map(convert, xml_paths)
    else:
        for xml_path in xml_paths:
            convert(xml_path)

if __name__ == '__main__':
    xml2csv()