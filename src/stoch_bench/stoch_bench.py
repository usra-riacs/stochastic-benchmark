# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def add_one(number):
    return number + 1

# %%
def BER(N,p,eng_list):
    '''
    Function to compute Bernoulli distribution from N datapoints using 
    probabilities p
    
    Return:
        ber (float): 
        ber=eng_list*(\sum_{k=0,...,n-1} \sum_{j=k,...,n}{p_{j}}^N - \sum_{j=k+1,...,n}{p_{j}}^N
            + p_n^N )
    Arguments:
        N (int): Number of data points
        p (np.array): Array of probabilities
        eng_list (np.array): List of energies
    '''
    ber = 0
    p = np.array(p)
    
    for k in range(len(p)):
        if k+1< len(p):
            ber += (np.sum(p[k:])**N - np.sum(p[k+1:])**N)*eng_list[k]
        else:
            ber += p[k]**N * eng_list[k]
    return np.round(ber,8)

def cumulative(energy, p, eng_list):
    '''
    Function to return whether a probability distribution associated with a list of 
    energy is beyond a threshold
    
    Return:
        1 if value is below the threshold, 0 otherwise
    Arguments:
        energy (float): Energy threshold
        p (np.array): Array of probabilities
        eng_list (np.array): List of energies
    '''
    if min(eng_list) > energy:
        return 0
    elif max(eng_list) < energy:
        return 1
    else:
        cumulative_dist = np.cumsum(p)
        for i, E in enumerate(eng_list): 
            if E > energy:
                return cumulative_dist[i-1]
    return 1

def confidence_prob(N, p, eng_list, eng_th):
    '''
    Return the confidence probability based on the cumulative
    
    Return:
        1-(1-F)^N
    Arguments:
        N (int): Number of data points
        p (np.array): Array of probabilities
        eng_list (np.array): List of energies
        eng_th (float): Energy threshold
    '''
    F = cumulative(eng_th , p, eng_list)
    
    return 1-(1-F)**N

def invCDF(N, prob, eng_list, conf = 0.99):
    '''
    Return the inverse of the cumulative distribution function
    
    Return:
        eng_list[idx*] (float):
            energy such that its index corresponds to certain confidence interval
    Arguments:
        N (int): Number of data points
        p (np.array): Array of probabilities
        eng_list (np.array): List of energies
        conf (float) = 0.99: Confidence threshold
    '''
    cdf = np.cumsum(prob)    
    threshold = 1 - (1-conf)**(1/N)
    for idx, p in enumerate(cdf):
        
        if p >= threshold:
            if idx ==0:
                return eng_list[0]
            else:
                return eng_list[idx]
            
def failure_prob(N, p, eng_list, eng_th):
    '''
    Return the failure probability based on the cumulative
    
    Return:
        (1-F)^N
    Arguments:
        N (int): Number of data points
        p (np.array): Array of probabilities
        eng_list (np.array): List of energies
        eng_th (float): Energy threshold
    '''
    F = cumulative(eng_th , p, eng_list)
    
    return (1-F)**N

def runs(Eth, conf, prob, eng_list ):
    '''
    Return the estimate of runs based on the cumulative
    
    Return:
        log(1-conf)/log(1-F)
    Arguments:
        Eth (float): Energy threshold
        conf (float): Confidence threshold for which the numer of runs are evaluated
        prob (np.array): Array of probabilities
        eng_list (np.array): List of energies
    '''
    F = cumulative(Eth, prob, eng_list)
    if F == 1:
        return 1
    else:
        return np.log(1-conf)/np.log(1-F)

# %%

def plot_avg_pass(n_data, method="avg", save_fig=True):
    '''
    Returns a figure for the average passes of E vs N
    
    Return:
        fig (matplotlib figure): FIgure with all the 
    Arguments:
        n_data (int): Number of datapoints to be plotted
        method = "avg" (str): String of instances to be plotted
        save_fig = True (bool): Boolean to save figure
    '''
    df_total_file = "../code_for_art/normalized_prob_dist/QPSK/" + \
        method + "/all_avg/avg_pass_test.csv"
    df_total = pd.read_csv(df_total_file, index_col=0)
    if n_data is None:
        n_data = 1000  # Number of datapoints
    elif n_data > df_total.shape[0]:
        print("n_data is larger than " +
              str(df_total.shape[0]) + " available datapoints, setting it to maximum")
        n_data = df_total.shape[0]

    N_runs_list = list(range(1, n_data+1))

    fig, ax = plt.subplots()
    fig.set_figheight(12)
    fig.set_figwidth(24)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=25)

    for column in df_total:
        ax.plot(N_runs_list,
                df_total[column].values[:n_data], c="grey", alpha=0.5)
    ax.plot(N_runs_list, df_total["avg"][:n_data], c="r", label="Mean")

    ax.set_xlabel('#N', fontsize=25)
    ax.set_ylabel("E",  fontsize=25)
    fig.legend(loc=(0.9, 0.9))

    if save_fig:
        path_plots = "new_plots/" + method + "/runs_pass_test/"
        if not os.path.exists(path_plots):
            os.makedirs(path_plots)

        title = "pass_test_NQ=" + str(44) + "_total_1.pdf"

        fig.savefig(path_plots + title + ".pdf", bbox_inches='tight')

    plt.close(fig)
    return fig

# %%
plot_avg_pass(100)
# %%
