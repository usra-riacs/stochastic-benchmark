import os


class paths:
    def __init__(self, home_dir):
        self.cwd = home_dir
        self.raw_data = os.path.join(self.cwd, 'exp_raw')
        self.checkpoints = os.path.join(self.cwd, 'checkpoints')
        self.plots = os.path.join(self.cwd, 'plots')

        for path in [self.checkpoints, self.plots]:
            if not os.path.exists(path):
                os.makedirs(path)

    def bootstrap(self):
        return os.path.join(self.checkpoints, 'bootstrap_results.pkl')

        #     def param_path(self, base, param_dict, ignore = []):
#         if case == 'results':
#             ppath = os.path.join(self.results, param2filename(param_dict, '', ignore))
#         elif case == 'data':
#             ppath = os.path.join(self.data, param2filename(param_dict, '', ignore))
#         elif case == 'plots':
#             ppath = os.path.join(self.plots, param2filename(param_dict, '', ignore))
#         elif case == 'instances':
#             ppath = os.path.join(self.instances, param2filename(param_dict, '', ignore))
#         elif case is None:
#             ppath = os.path.join(self.cwd, param2filename(param_dict, '', ignore))


#         if not os.path.exists(ppath):
#             os.makedirs(ppath)
#         return ppath


def param2filename(param_dict, ext, ignore=[]):
    """
    Utility to turn parameter dictionary into filename

    Parameters
    ----------
    param_dict : dict
        dictionary of parameters
    ext : str
        extension of filename
    ignore : list
        list of parameters to ignore

    Returns
    -------
    filename : str
        filename
    """
    def val2str(v): return str(v)  # update this for better float formatting
    filename = ['{}={}_'.format(str(k), val2str(param_dict[k])) for
                k in sorted(param_dict.keys()) if
                k not in ignore]
    filename = ''.join(filename)
    filename = filename[:-1] + ext

    return filename


def filename2param(filename):
    """
    Utility to turn filename into parameter dictionary, user needs to fix types of values

    Parameters
    ----------
    filename : str
        filename

    Returns
    -------
    param_dict : dict
        dictionary of parameters
    """
    # utility to turn filename into parameter dictionary, user needs to fix types of values
    split_filename = filename.split('_')
    ext_idx = split_filename[-1].rindex('.')
    split_filename[-1] = split_filename[-1][:ext_idx]

    param_dict = {}
    for kv in split_filename:
        k, v = kv.split('=')
        param_dict[k] = v
    return param_dict
