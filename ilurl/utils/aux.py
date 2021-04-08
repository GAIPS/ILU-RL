__date__ = '2020-06-08'
import re
from collections import Iterable

PATTERN = re.compile(r'(?<!^)(?=[A-Z])')
TIMESTAMP = re.compile(r'\d{8}-\d{8,16}.\d{1,7}')

def camelize(snake_case_name):
    """Converts from SnakeCase to camel_case.

    Params:
    -------
    snake_case_name: str
        ThisIsASnakeCaseName

    Returns:
    -------
    camel_case_name: str
        this_is_a_camel_case_name in `_' format
    """
    return PATTERN.sub('_', snake_case_name).lower()

def snakefy(word):
    """Converts from camel_case SnakeCase.

    Params:
    -------
    word: str
        this_is_a_camel_case_name in `_' format

    Returns:
    -------
    word: str
        ThisIsASnakeCaseName
    """
    return ''.join(w.title() for w in word.split('_'))

def flatten(items, ignore_types=(str, bytes)):
    """

    Usage:
    -----
    > items = [1, 2, [3, 4, [5, 6], 7], 8]

    > # Produces 1 2 3 4 5 6 7 8
    > for x in flatten(items):
    >         print(x)

    Ref:
    ----

    David Beazley. `Python Cookbook.'
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x

class Printable(object):
    def __repr__(self):
        """Returns a string containing the attributes of the class."""
        text_repr = f"\n{self.__class__.__name__}:\n"
        for (attr, val) in self.__dict__.items():
            text_repr += f"{attr}: {val}\n"
        return text_repr

