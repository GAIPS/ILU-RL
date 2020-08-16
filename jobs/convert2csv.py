from pathlib import Path
from ilurl.loaders.xml2csv import main as xml2csv

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str, nargs='?',
                        help='''A directory which it\'s subdirectories are train runs.''')

    parsed = parser.parse_args()
    sys.argv = [sys.argv[0]]

    return parsed

if __name__ == '__main__':

    # Parse command line.
    args = get_arguments()
    experiment_root_path = args.experiment_dir

    print('\nConverting .xml files to .csv ...\n')

    # Convert all .xml files to .csv format.
    for xml_path in Path(experiment_root_path).rglob('*.xml'):
        csv_path = str(xml_path).replace('xml', 'csv')
        args = [str(xml_path), '-o', csv_path]
        try:
            xml2csv(args)
            Path(xml_path).unlink()
        except Exception:
            raise