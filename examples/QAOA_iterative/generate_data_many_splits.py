# Given some bootstrapped data, implement interpolation followed by experiments (such as projection, sequential and random search experiments) for a specified number of test-train splits.
# Expects a folder named checkpoints with the bootstrapped data in pickle files.
# The script stored output for split #n in a folder named checkpoints_n
# Also stores plots for each individial split in the plots folder
import argparse
import os
import re
import shutil

import sys
sys.path.append('../../src')
import stochastic_benchmark as SB
import os
import bootstrap
import interpolate
import stats
from utils_ws import *
import plotting

dir_path = os.path.dirname(os.path.realpath(__file__)) # Path of current file

# The data should be available in the form of pkl files containing bootstrapped data in a checkpoints folder.
def single_split(split_ind, train_test_split = 0.8):
    """Implement analysis for a single test-train split

    Args:
        split_ind (int)
        train_test_split (float): If 0.8, 80% of instaces will be labeled train, and the remaining will be test
    """
    here = dir_path #os.getcwd()
    parameter_names = ['iterations', 'shots', 'rounds']
    instance_cols = ['instance']

    ## Response information 
    response_key = 'approx_ratio'
    response_dir = 1 # whether we want to maximize (1) or minimize (-1)

    ## Optimizations informations
    recover = True #Whether we want to read dataframes when available, default is True
    reduce_mem = True #Whether we want to segment bootstrapping and interpolation to reduce memory usage, default is True
    smooth = True  #Whether virtual best should be monontonized, default is True

    sb = SB.stochastic_benchmark(parameter_names=parameter_names, here=here, instance_cols=instance_cols, response_key=response_key, response_dir=response_dir, smooth=smooth)

    # Load Bootstrap data. 
    # The data is already boostrapped, but needs to be loaded into memory
    shared_args = {'response_col':"approx_ratio",
                'resource_col':"resource",
                'response_dir':1,
                'confidence_level':68}
    boots_range = [1,2,5, 10, 20, 50, 100]
    bsParams = bootstrap.BootstrapParameters(shared_args=shared_args, update_rule= lambda df: None)
    bs_iter_class = bootstrap.BSParams_range_iter()
    bsParams_iter = bs_iter_class(bsParams, boots_range)
    sb.run_Bootstrap(bsParams_iter)


    # Interpolate
    def resource_fcn(df):
        return df['boots'] * df['iterations'] * df['shots']
    iParams = interpolate.InterpolationParameters(resource_fcn,
                                                parameters=parameter_names)
    sb.run_Interpolate(iParams)


    # Set up Stats computations
    # train_test_split = 0.8
    metrics = ["approx_ratio"]
    stParams = stats.StatsParameters(metrics=metrics, stats_measures=[stats.Median()])
    sb.run_Stats(stParams, train_test_split)

    sb.run_baseline()
    sb.run_ProjectionExperiment('TrainingStats', None, None)
    sb.run_ProjectionExperiment('TrainingResults', None, None)
    
    
    plotting.monotone = True
    sb.initPlotting()
    plot_folder = os.path.join(here, 'plots', 'split_ind={}'.format(split_ind))
    if not os.path.exists(plot_folder): os.makedirs(plot_folder)
    fig, axs = sb.plots.plot_performance()
    fig.savefig(os.path.join(plot_folder,'performance.png')); fig.savefig(os.path.join(plot_folder,'performance.pdf'))
    figs, axes = sb.plots.plot_parameters_separate()
    for param, fig in figs.items():
        fig.savefig(os.path.join(plot_folder,param+'.png')); fig.savefig(os.path.join(plot_folder,param+'.pdf'))
    fig, axes = sb.plots.plot_parameters_together()
    fig.savefig(os.path.join(plot_folder,'all_params.png')); fig.savefig(os.path.join(plot_folder,'all_params.pdf'))
    
    del sb # helps in closing all files opened by pandas and pickle
    return 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ns', type=int, help="number of splits", default=10)
    parser.add_argument('-tt', type=float, help="Test-train split.", default=0.8)
    
    args = parser.parse_args()
    num_splits = args.ns

    for split_ind in range(num_splits):
        
        folder_1 = os.path.join(dir_path, 'checkpoints_{}'.format(split_ind))
        folder_2 = os.path.join(dir_path,'checkpoints')
        # if os.path.exists(folder_1):
        #     # If processed partially, it will be useful to run stochastic_benchmark again
        #     # if os.path.exists(folder_2): os.rmdir(folder_2)
        #     os.rename(folder_1, folder_2) # src, dest
        # Analyze a single test-train split
        single_split(split_ind, train_test_split = args.tt)
        
        # Rename checkpoints folder to be checkpoints_split_ind
        
        os.rename(folder_2, folder_1)
        
        # If this is not the last split_ind create a new checkpoints folder
        if split_ind != num_splits - 1:
            # create new folder called checkpoints
            os.mkdir(folder_2)
            # copy all bootstrap files and interp results file from checkpoints_{} to checkpoints
            list_of_files = os.listdir(folder_1)
            pattern = 'bootstrapped_results_inst=([0-9]+).pkl'
            boots_files_list = [fileName for fileName in list_of_files if re.search(pattern, fileName)]
            
            for boots_file_name in boots_files_list:
                shutil.copy(os.path.join(folder_1, boots_file_name), os.path.join(folder_2, boots_file_name))  #source, destination
            # also copy interp_results pkl file, but delete train column
            df = pd.read_pickle(os.path.join(folder_1, 'interpolated_results.pkl'))
            del df['train']
            pd.to_pickle(df, os.path.join(folder_2, 'interpolated_results.pkl'))
            del df