import os


class paths:
    """
    Class to define paths to various files
    """
    def __init__(self, home_dir):
        self.cwd = home_dir
        self.raw_data = os.path.join(self.cwd, 'exp_raw')
        self.checkpoints = os.path.join(self.cwd, 'checkpoints')
        self.plots = os.path.join(self.cwd, 'plots')
        self.progress = os.path.join(self.cwd, 'progress')

        for path in [self.checkpoints, self.plots, self.progress]:
            if not os.path.exists(path):
                os.makedirs(path)
        self.bootstrap = os.path.join(self.checkpoints, 'bootstrapped_results.pkl')
        self.interpolate = os.path.join(self.checkpoints, 'interpolated_results.pkl')
        self.training_stats = os.path.join(self.checkpoints, 'training_stats.pkl')
        self.testing_stats = os.path.join(self.checkpoints, 'testing_stats.pkl')
        
        self.virtual_best = {'train': os.path.join(self.checkpoints, 'vb_train.pkl'),\
                             'test': os.path.join(self.checkpoints, 'vb_test.pkl')}
        
        self.best_rec = {'stats': os.path.join(self.checkpoints, 'br_stats.pkl'),\
                         'results': os.path.join(self.checkpoints, 'br_results.pkl')}
        
        self.projections = {'stats': os.path.join(self.checkpoints, 'proj_stats.pkl'),\
                             'results': os.path.join(self.checkpoints, 'proj_results.pkl')}
        
        self.best_agg_alloc = os.path.join(self.checkpoints, 'best_agg_alloc.pkl')
        self.train_exp_at_best = os.path.join(self.checkpoints, 'train_exp_at_best.pkl')
        self.final_values = os.path.join(self.checkpoints, 'final_values.pkl')
        self.test_exp_at_best = os.path.join(self.checkpoints, 'test_exp_at_best.pkl')
        self.seq_exp_values = os.path.join(self.checkpoints, 'seq_exp_values.pkl')

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
    if '.' in split_filename[-1]:
        ext_idx = split_filename[-1].rindex('.')
        split_filename[-1] = split_filename[-1][:ext_idx]

    param_dict = {}
    for kv in split_filename:
        k, v = kv.split('=')
        param_dict[k] = v
    return param_dict
