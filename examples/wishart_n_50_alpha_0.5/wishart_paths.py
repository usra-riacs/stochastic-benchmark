import os


alpha = '0.50'

# Use relative paths based on this script's location.
script_dir = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.join(script_dir, 'data')
rerun_datapath = os.path.join(script_dir, 'rerun_data')


def logname(instance_num, sweeps, replicas, pcold, phot, rerun=False):
    log = 'inst={}_pcold={:.2f}_phot={:.1f}_replicas={}_sweeps={}.pkl'.format(
        instance_num, pcold, phot, replicas, sweeps)
    obj = 'obj_inst={}_pcold={:.2f}_phot={:.1f}_replicas={}_sweeps={}.pkl'.format(
        instance_num, pcold, phot, replicas, sweeps)
    if rerun:
        return os.path.join(rerun_datapath, log), os.path.join(rerun_datapath, obj)
    return os.path.join(datapath, log), os.path.join(datapath, obj)
