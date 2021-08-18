import inspect
import json
import numpy as np

from fire import Fire
from typing import Union


def to_console(f):
    sig = inspect.getfullargspec(f)
    convert = [i for i, x in enumerate(sig.annotations) if sig.annotations[x] == np.ndarray]
    convert_ret = sig.annotations.get('return', None) == np.ndarray
    def new_f(*args, **kwargs):
        new_args = [json_to_np(x) if i in convert else x for i, x in enumerate(args)]
        ret = f(*new_args, **kwargs)
        if convert_ret:
            ret = np_to_json(ret)
        print(ret)
    return new_f


def json_to_np(json_array: Union[dict, np.ndarray]) -> np.ndarray:
    
    if isinstance(json_array, np.ndarray):
        return json
    else: 
        array = np.array(json_array['data'], dtype=json_array['dtype'])
        return array


def np_to_json(array: np.ndarray) -> dict:
    d = {
        "data": array.tolist(),
        "shape": list(array.shape),
        "dtype": array.dtype.name
    }
    return json.dumps(d)


@to_console
def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


@to_console
def hello():
    print("Hello")


if __name__ == "__main__":
    Fire()