"""More beautiful JSON encoder than the default, modified by Simon Ging from:
https://stackoverflow.com/questions/10097477/python-json-array-newlines
"""
import collections
import numpy as np

INDENT = 3
SPACE = " "
NEWLINE = "\n"


class StringHandler(object):
    def __init__(self, fh=None):
        self.str_coll = []
        self.fh = fh

    def __add__(self, other):
        if self.fh is not None:
            self.fh.write(other)
        else:
            self.str_coll.append(other)
        return self

    def get_return_value(self):
        if self.fh is not None:
            return ""
        else:
            return "".join(self.str_coll)


def to_json(o, sort_keys=False):
    return _read_json(o, sort_keys=sort_keys)


def write_json(o, fh, sort_keys=False):
    _read_json(o, fh=fh, sort_keys=sort_keys)


def _read_json(o, level=0, fh=None, sort_keys=False):
    int_classes = [int, np.int, np.long, np.int8, np.int16, np.int32, np.int64]
    float_classes = [
        float, np.float, np.double, np.half, np.float16, np.float32, np.float64
    ]
    ret = StringHandler(fh=fh)
    if isinstance(o, collections.Mapping):
        ret += "{" + NEWLINE
        comma = ""
        if sort_keys:
            keys = sorted(o.keys())
        else:
            keys = o.keys()
        for k in keys:
            v = o[k]
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += _read_json(v, level + 1, fh=fh, sort_keys=sort_keys)
        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        # check str before iterable, since str is an iterable as well
        # also escaped some of the escaped characters that will ruin the
        # json format
        o = o.replace("\n", "\\n")
        o = o.replace("\"", "\\\"")
        ret += '"' + o + '"'
    elif isinstance(o, collections.Iterable):
        ret += "["
        comma = ""
        for e in o:
            ret += comma
            comma = ","
            ret += _read_json(e, level + 1, fh=fh, sort_keys=sort_keys)
        ret += "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif np.any([isinstance(o, a) for a in int_classes]):
        ret += str(o)
    elif np.any([isinstance(o, a) for a in float_classes]):
        ret += str(o)  # '%.7g' % o
    elif isinstance(o, np.ndarray):
        if np.issubdtype(o.dtype, np.integer):
            ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
        elif np.issubdtype(o.dtype, np.inexact):
            ret += "[" + ','.join(
                map(lambda x: '%.7g' % x,
                    o.flatten().tolist())) + "]"
        else:
            raise TypeError("unknown np dtype {}".format(o.dtype))
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" %
                        str(type(o)))
    return ret.get_return_value()
