from timeit import default_timer as timer


def timeit_decorator(method):
    """
    functions with this decorator will print their execution time by default.
    if a dictionary is passed to kwarg log_time, the time will instead be saved
    to that dictionary, either with key from kwarg log_name or by default the
    method name as key.

    Args:
        method: function to decorate

    Returns:
        decorated function

    """
    def timed(*args, **kw):
        ts = timer()
        result = method(*args, **kw)
        te = timer()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print("method: {} time: {:.3f}s".format(method.__name__,
                                                    (te - ts)))
        return result

    return timed


class EasyTimer(object):
    def __init__(self):
        self.start_time = 0
        self.reset()

    def reset(self):
        self.start_time = timer()

    def __call__(self, add_info="", print_fn=print, reset=False):
        diff_seconds = timer() - self.start_time
        str_ = ""
        if add_info != "":
            str_ = "{} ".format(add_info)

        if diff_seconds >= 300:
            print_fn("{}{:.3f}min".format(str_, diff_seconds / 60))
        else:
            print_fn("{}{:.3f}sec".format(str_, diff_seconds))
        if reset:
            self.reset()
        return self
