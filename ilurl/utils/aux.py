__date__ = '2020-06-08'
import re

PATTERN = re.compile(r'(?<!^)(?=[A-Z])')


def camelize(snake_case_name):
    """Converts from SnakeCase to camel_case.

    Params:
    -------
    snake_case_name: str
        a name to be converted

    Returns:
    -------
    camel_case_name: str
        a name in `_' format
    """
    return PATTERN.sub('_', snake_case_name).lower()


class Printable(object):
    def __repr__(self):
        """Returns a string containing the attributes of the class."""
        text_repr = f"\n{self.__class__.__name__}:\n"
        for (attr, val) in self.__dict__.items():
            text_repr += f"{attr}: {val}\n"
        return text_repr