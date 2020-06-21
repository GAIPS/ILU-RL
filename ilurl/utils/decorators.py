import sys
import json
from time import time
from pathlib import Path

from ilurl.utils.context_managers import PipeGuard


def benchmarked(fnc):
    """Times execution of fnc, storing on folder if path exists.

        Parameters:
        ----------
        * fnc: function
            An anonymous function decorated by the user.

        Returns:
        -------
        * f: function
            An anonymous function that will be timed.
    """
    _data = {}
    def f(*args, **kwargs):
        _data['start'] = time()
        res = fnc(*args, **kwargs)
        _data['finish'] = time()
        _data['elapsed'] = _data['finish'] - _data['start']

        # if res is a valid sys path
        if Path(str(res)).exists():
            target_path = Path(str(res)) / 'time.json'
            with target_path.open('w') as f:
                json.dump(_data, f)
        else:
            print(f'''\tChronological:
                        ---------------
                      \t start:{_data['start']}
                      \t finish:{_data['finish']}
                      \t elapsed:{_data['elapsed']}\n\n''')
        return res
    return f


def processable(fnc):
    """Supresses stdout during fnc execution writing output only.

        Parameters:
        ----------
        * fnc: function
            An anonymous function decorated by the user.

        Returns:
        -------
        * f: function
            An anonymous function that will have stdout supressed.
    """
    def f(*args, **kwargs):
        with PipeGuard():
            res = fnc(*args, **kwargs)
        # send result to the pipeline
        sys.stdout.write(res)
        return res
    return f


def safe_run(func, error_message=None):
    """Wraps function within a try catch block.

        Parameters:
        ----------
        * error_message: str
            Error message to be displayed if an error is caught.

    """
    def func_wrapper(*args, **kwargs):

        try:
           return func(*args, **kwargs)
        except Exception as e:
            print(e)
            if error_message:
                print(error_message)
            return None

    return func_wrapper