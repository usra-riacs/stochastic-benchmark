import dill
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.fmin import generate_trials_to_calculate
import math
import numpy as np
import os
import pandas as pd
try:
    from pysa.sa import Solver
except:
    print('no pysa found')
from scipy.sparse import csr_matrix
import sys
from tqdm import tqdm
import time

N = 50
alpha = '0.50'
n_reads = 1001 #TODO change this if you want
float_type = 'float32'
penalty = 1e6
datapath = '/home/bernalde/repos/stochastic-benchmark/examples/wishart_N=50_alpha={}/data'.format(alpha) #TODO change the directory where you want this
rerun_datapath = '/home/bernalde/repos/stochastic-benchmark/examples/wishart_N=50_alpha={}/rerun_data'.format(alpha) #TODO change the directory where you want this

class seen_result:
    def __init__(self):
        self.obj = np.inf
        self.idx_list = []
        self.runtime_list = []

    def to_dateframe(self):
        return

def logname(instance_num, sweeps, replicas, pcold, phot, rerun=False):
    log = 'inst={}_pcold={:.2f}_phot={:.1f}_replicas={}_sweeps={}.pkl'.format(instance_num, pcold, phot, replicas, sweeps)
    obj = 'obj_inst={}_pcold={:.2f}_phot={:.1f}_replicas={}_sweeps={}.pkl'.format(instance_num, pcold, phot, replicas, sweeps) 
    if rerun:
        return os.path.join(rerun_datapath, log), os.path.join(rerun_datapath, obj)
    else:
        return os.path.join(datapath, log), os.path.join(datapath, obj)

def sol_to_bitstring(sol):
    bitstring = ''.join([str(int(x)) for x in sol])
    return bitstring

def seen_to_df(seen):
    record = []
    for k, v in seen.items():
        record.append((k, v.obj, 1e-6 * np.mean(v.runtime_list), len(v.idx_list)))
    df = pd.DataFrame.from_records(record, columns = ['SolString','Energy', 'MeanTime', 'count'])
    return df

def obj_fcn(norm_score, mean_time, replicas, s):
    # TODO double check this objective
    if norm_score == 1.:
        return replicas * 1e-6 * mean_time
    elif norm_score == 0.:
        return penalty
    else:
        return replicas * 1e-6 * mean_time * np.log(1 - s) / np.log(1 - norm_score)

def load_instance(instance_num):
    #TODO fill in your correct path here
    base_dir = '/home/bernalde/repos/stochastic-benchmark-backup/data/wishart/instance_generation/wishart_planting_N_50_alpha_{}'.format(alpha)
    inst_name = 'wishart_planting_N_50_alpha_{}_inst_{}.txt'.format(alpha, instance_num)
    filename = os.path.join(base_dir, inst_name)
    
    rows = []
    cols = []
    vals = []
    
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            line = line.strip().split('\t')
            rows.append(int(line[0]))
            cols.append(int(line[1]))
            vals.append(int(line[2]))
            line = f.readline()

    qubo = csr_matrix((vals, (rows, cols)), shape = (N, N))

    gs_filename = os.path.join(base_dir, 'gs_energies.txt')
    gs_dict = {}
    with open(gs_filename) as f:
        line = f.readline()
        while line:
            line = line.strip().split('\t')
            gs_dict[line[0]] = float(line[1])
            line = f.readline()
    gs_energy = gs_dict[inst_name]
    return (qubo + qubo.T)/2., gs_energy

def run_pysa(args, instance_num, pbar=None):
    sweeps = int(args['sweeps'])
    replicas = int(args['replicas'])
    pcold = np.round(float(args['pcold']), decimals=2)
    phot = np.max(0.1, np.round(float(args['phot']), decimals=1))

    qubo, gs_energy = load_instance(instance_num)

    df_filename, obj_filename = logname(instance_num, sweeps, replicas, pcold, phot)
    if os.path.exists(obj_filename):
        try:
            with open(obj_filename, 'rb') as f:
                norm_score = dill.load(f)
                mean_time = dill.load(f)
            
            obj = obj_fcn(norm_score, mean_time, replicas, 0.99)
            
            return obj
        except:
            print('rerunning pysa')
            #TODO double check these!!!
            min_temp = 2 * np.min(np.abs(qubo[np.nonzero(qubo)])) / np.log(100/pcold)
    #         min_temp_cal = 2*min(sum(abs(i) for i in qubo)) / np.log(100/p_cold)
            max_temp = 2*max(sum(abs(i) for i in qubo.A)) / np.log(100/phot)
            solver = Solver(problem=qubo.A, problem_type='ising', float_type=float_type)
            res = solver.metropolis_update(
                num_sweeps = sweeps,
                num_reads = n_reads,
                num_replicas = replicas,
                update_strategy='random',
                min_temp = min_temp,
                max_temp = max_temp,
                recompute_energy=True,
                sort_output_temps=True,
                parallel=True,
                use_pt=True,
                verbose=False)
            norm_score, mean_time, seen = process_pysa(res, gs_energy)
            with open(obj_filename, 'wb') as f:
                dill.dump(norm_score, f, dill.HIGHEST_PROTOCOL)
                dill.dump(norm_score, f, dill.HIGHEST_PROTOCOL)
    else:
        min_temp = 2 * np.min(np.abs(qubo[np.nonzero(qubo)])) / np.log(100/pcold)
#         min_temp_cal = 2*min(sum(abs(i) for i in qubo)) / np.log(100/p_cold)
        max_temp = 2*max(sum(abs(i) for i in qubo.A)) / np.log(100/phot)
        solver = Solver(problem=qubo.A, problem_type='ising', float_type=float_type)
        res = solver.metropolis_update(
            num_sweeps = sweeps,
            num_reads = n_reads,
            num_replicas = replicas,
            update_strategy='random',
            min_temp = min_temp,
            max_temp = max_temp,
            recompute_energy=True,
            sort_output_temps=True,
            parallel=True,
            use_pt=True,
            verbose=False)

        norm_score, mean_time, seen = process_pysa(res, gs_energy)
        with open(obj_filename, 'wb') as f:
            dill.dump(norm_score, f, dill.HIGHEST_PROTOCOL)
            dill.dump(mean_time, f, dill.HIGHEST_PROTOCOL)
    seen_df = seen_to_df(seen)
    seen_df['sweeps'] = sweeps
    seen_df['replicas'] = replicas
    seen_df['pcold'] = pcold
    seen_df['phot'] = phot
    seen_df['GTMinEnergy'] = gs_energy
    
    seen_df.to_pickle(df_filename)
    obj = obj_fcn(norm_score, mean_time, replicas, 0.99)
    if pbar is not None:
        pbar.update()
    return obj

def rerun_pysa(params, instance_num):
    sweeps = int(params[1])
    replicas = int(params[2])
    pcold = np.round(params[3], decimals=2)
    phot = np.maximum(0.1, np.round(params[4], decimals=1))

    qubo, gs_energy = load_instance(instance_num)

    df_filename, obj_filename = logname(instance_num, sweeps, replicas, pcold, phot, rerun=True)
    if os.path.exists(df_filename) and os.path.exists(obj_filename):
        seen_df = pd.read_pickle(df_filename)
        # print('reading from file')
        return seen_df

    else:
        #TODO double check these!!!
        print('Trying to run pysa for parameters ', params)
        min_temp = 2 * np.min(np.abs(qubo[np.nonzero(qubo)])) / np.log(100/pcold)
#         min_temp_cal = 2*min(sum(abs(i) for i in qubo)) / np.log(100/p_cold)
        max_temp = 2*max(sum(abs(i) for i in qubo.A)) / np.log(100/phot)
        solver = Solver(problem=qubo.A, problem_type='ising', float_type=float_type)
        res = solver.metropolis_update(
            num_sweeps = sweeps,
            num_reads = n_reads,
            num_replicas = replicas,
            update_strategy='random',
            min_temp = min_temp,
            max_temp = max_temp,
            recompute_energy=True,
            sort_output_temps=True,
            parallel=True,
            use_pt=True,
            verbose=False)
        norm_score, mean_time, seen = process_pysa(res, gs_energy)
        with open(obj_filename, 'wb') as f:
            dill.dump(norm_score, f, dill.HIGHEST_PROTOCOL)
            dill.dump(mean_time, f, dill.HIGHEST_PROTOCOL)

    seen_df = seen_to_df(seen)
    seen_df['sweeps'] = sweeps
    seen_df['replicas'] = replicas
    seen_df['pcold'] = pcold
    seen_df['phot'] = phot
    seen_df['GTMinEnergy'] = gs_energy
    seen_df.to_pickle(df_filename)

    return seen_df


def process_pysa(res, gs_energy):
    runtimes = res["runtime (us)"][1:]
    mean_time = np.mean(runtimes)
    seen = {}
    for read_num in range(1, n_reads):
        pysa_sol = res.best_state[read_num]
        pysa_obj = 2*res.best_energy[read_num]
        bit_pysa_sol = sol_to_bitstring(pysa_sol)
        
        if bit_pysa_sol in seen:
            seen[bit_pysa_sol].idx_list.append(read_num)
            seen[bit_pysa_sol].runtime_list.append(res['runtime (us)'][read_num])
        else:
            seen[bit_pysa_sol] = seen_result()
            seen[bit_pysa_sol].obj = pysa_obj
            seen[bit_pysa_sol].idx_list = [read_num]
            seen[bit_pysa_sol].runtime_list = [res['runtime (us)'][read_num]]
    
    #TODO check the objective
    scores = []
    for v in seen.values():
        score = 1. - (min(v.obj, 0.) - gs_energy) / abs(gs_energy)
        scores.extend(len(v.idx_list) * [score])
    norm_score = np.mean(scores)
    return norm_score, mean_time, seen

def run_hyperopt(h, hpo_trial, instance_num):
    # TODO set your parameters space
    spaceVar = {'sweeps': hp.qloguniform('sweeps', 0, 4, 1),
                'replicas': hp.quniform('replicas', 1, 16, 1),
                'pcold': hp.lognormal('pcold', 0, 0.25),
                'phot': hp.uniform('phot', 0., 100.)}
    
    # TODO set grid params
    if h == 1:
        grid_params = []
        replicas_list = [32,24,16,12,8,6,4,2,1]
        sweeps_list = [1,2,5,7,10,15,20,30,50,70,100,150,200,300,500]
        # sweeps_list = [1]
        phot_list = [10,30,50,70,90]
        for replica in replicas_list:
            for sweep in sweeps_list:
                for phot in phot_list:
                    grid_params.append({'sweeps':sweep, 'replicas':replica, 'pcold':1, 'phot':phot})
        # grid_params = [{'sweeps':10, 'replicas':1, 'pcold':1, 'phot':50}]
        trials = generate_trials_to_calculate(grid_params)
        total = len(grid_params) + 100
        
    else:
        trials = generate_trials_to_calculate([])
        total = 100
    
    # Call hyperopt
    # TODO check max_evals
    pbar = tqdm(total=total, desc="Hyperopt")
    objective = lambda args : run_pysa(args, instance_num, pbar)
    best = fmin(fn = objective,
                space=spaceVar,
                algo=tpe.suggest,
                max_evals = 100,
                trials=trials)

    hpo_trials_name = os.path.join(datapath, 'hpoTrials_warmstart={}_trial={}_inst={}.pkl'.format(h, hpo_trial, instance_num))
    with open(hpo_trials_name , 'wb') as f:
        dill.dump(trials, f, dill.HIGHEST_PROTOCOL)
    pbar.close()
    return

def rerun_outer(instance_num):
    for rerun_params_filename in ['rerun_params_0.txt', 'rerun_params_1.txt']:
        rerun_params = np.loadtxt(rerun_params_filename)
        for idx in range(rerun_params.shape[0]):
            params = rerun_params[idx, :]
            rerun_pysa(params, instance_num)

if __name__ == '__main__':
    instance_num = int(os.getenv('PBS_ARRAY_INDEX'))
    rerun_outer(instance_num)

    # #base_num = 500
    # n_insts = 50
    # n_hpo = 10
    # h = math.floor(base_num / (n_hpo * n_insts))
    # remainder = base_num % (n_hpo * n_insts)
    # hpo_trial = math.floor(remainder / n_insts)
    # instance_num = (remainder % n_insts) + 1

    # #instance_num = int(os.getenv('PBS_ARRAY_INDEX'))
    # run_hyperopt(h, hpo_trial, instance_num)
