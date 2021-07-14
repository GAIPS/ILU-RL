import numpy as np

import tree

from acme import types


def double_to_single_precision(nested_value: types.Nest) -> types.Nest:
    """Convert a nested value given a desired nested spec."""

    def _convert_single_value(value):
        if value is not None:
            value = np.array(value, copy=False)
            if np.issubdtype(value.dtype, np.float64):
                value = np.array(value, copy=False, dtype=np.float32)
            elif np.issubdtype(value.dtype, np.int64):
                value = np.array(value, copy=False, dtype=np.int32)
        return value

    return tree.map_structure(_convert_single_value, nested_value)

def action_to_double_precision(nested_value: types.Nest) -> types.Nest:
    """Convert a nested value given a desired nested spec."""

    def _convert_single_value(value):

        if value is None:
            return value

        value = np.array(value, copy=False)
        if np.issubdtype(value.dtype, np.float32):
            value = np.array(value, copy=False, dtype=np.float64)
        elif np.issubdtype(value.dtype, np.int32):
            value = np.array(value, copy=False, dtype=np.int64)

        return np.asscalar(value)

    return tree.map_structure(_convert_single_value, nested_value)
