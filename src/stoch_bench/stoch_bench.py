# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def add_one(
    number: int,
) -> int:
    '''
    Test function for pip package

    Return:
        number + 1

    Arguments:
        number (int): Number to be added
    '''
    return number + 1

# %%


def BER(
    N: int,
    p: np.array,
    eng_list: np.array,
) -> float:
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
        if k+1 < len(p):
            ber += (np.sum(p[k:])**N - np.sum(p[k+1:])**N)*eng_list[k]
        else:
            ber += p[k]**N * eng_list[k]
    return np.round(ber, 8)


def cumulative(
    energy: float,
    p: np.array,
    eng_list: np.array,
) -> float:
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


def confidence_prob(
    N: int,
    p: np.array,
    eng_list: np.array,
    eng_th: float,
) -> float:
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
    F = cumulative(eng_th, p, eng_list)

    return 1-(1-F)**N


def invCDF(
    N: int,
    prob: np.array,
    eng_list: np.array,
    conf: float = 0.99,
) -> float:
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
            if idx == 0:
                return eng_list[0]
            else:
                return eng_list[idx]


def failure_prob(
    N: int,
    p: np.array,
    eng_list: np.array,
    eng_th: float,
) -> float:
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
    F = cumulative(eng_th, p, eng_list)

    return (1-F)**N


def runs(
    Eth: float,
    conf: float,
    prob: np.array,
    eng_list: np.array,
) -> float:
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


def plot_avg_pass(
    NQ: int = 44,
    n_data: int = 1000,
    method: str = "avg",
    save_fig: bool = True,
) -> None:
    '''
    Returns a figure for the average passes of E metric vs R resources

    Return:
        fig (matplotlib figure): FIgure with all the... TODO
    Arguments:
        NQ (int): Number of qubits
        n_data (int): Number of datapoints to be plotted
        method (str)= "avg" (str): String of instances to be plotted
        save_fig (bool)= True (bool): Boolean to save figure
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
    # fig.set_figheight(12)
    # fig.set_figwidth(24)
    ax.tick_params(axis='y')  # , labelsize=25)
    ax.tick_params(axis='x')  # , labelsize=25)

    for column in df_total:
        ax.plot(N_runs_list,
                df_total[column].values[:n_data], c="grey", alpha=0.5)
    ax.plot(N_runs_list, df_total["avg"][:n_data], c="r", label="Mean")

    ax.set_xlabel('R')  # , fontsize=25)
    ax.set_ylabel("E")  # ,  fontsize=25)
    fig.legend(loc=(0.9, 0.9))

    if save_fig:
        path_plots = "new_plots/" + method + "/runs_pass_test/"
        if not os.path.exists(path_plots):
            os.makedirs(path_plots)

        title = "pass_test_NQ=" + str(NQ) + "_total_1.pdf"

        fig.savefig(path_plots + title + ".pdf", bbox_inches='tight')

    plt.close(fig)
    return fig


# %%
plot_avg_pass(
    NQ=100,
    save_fig=False,
)
# %%


def plot_avg_cumulatives(
    NQ: int = 44,
    probs: int = 20,
    conf: float = 0.9,
    N_runs: int = 1000,
    method: str = "avg",
    save_fig: bool = True
) -> None:
    '''
    Returns a figure for the cumulative passes of E vs N

    Return:
        fig (matplotlib figure): FIgure with all the... TODO
    Arguments:
        NQ (int): Number of qubits
        probs (int): Number of total problems considered
        conf (float): confidence level for plot
        N_runs (int): Number of runs for plot
        method = "avg" (str): String of instances to be plotted
        save_fig = True (bool): Boolean to save figure
    '''
    path = "../code_for_art/normalized_prob_dist/QPSK/" + \
        method + "/N=" + str(N_runs) + "/"

    fig, ax = plt.subplots()
    # fig.set_figheight(12)
    # fig.set_figwidth(24)
    ax.tick_params(axis='y')  # ,labelsize = 25)
    ax.tick_params(axis='x')  # ,labelsize = 25)

    avg_cumulative = np.zeros(N_runs + 1)

    for problem in range(1, probs + 1):
        df_cum = pd.read_csv(path + "cumulative_norm_NQ=" + str(NQ) + "_problem=" +
                             str(problem) + "_N=" + str(N_runs) + ".csv", index_col=0)
        ax.plot(df_cum["eng"], df_cum["cumulative"], c="grey", alpha=0.5)
        avg_cumulative += df_cum["cumulative"].values/probs

    ax.plot(df_cum["eng"], avg_cumulative, c="r", label="Mean")

    for idx, cumu in enumerate(avg_cumulative):
        if cumu >= conf:
            E_pass_test = df_cum["eng"][idx]
            break

    # Intersection of the confidence interval and the average cumulative
    ax.axhline(conf, xmin=0, xmax=1, c="navy", ls=":", linewidth=2.5, zorder=0)
    ax.axvline(E_pass_test, c="navy", ls=":", linewidth=2.5, zorder=0)

    ax.set_xlabel('E')  # , fontsize = 25)
    ax.set_ylabel("Probability")  # ,  fontsize = 25)
    # fig.legend(loc=(0.9,0.1))
    # plt.show()

    if save_fig:
        path_plots = "../code_for_art/new_plots/" + method + "/mean/"
        if not os.path.exists(path_plots):
            os.makedirs(path_plots)

        title = "cumulatives_NQ=" + str(NQ) + "mean_N=" + str(N_runs)
        fig.savefig(path_plots + title + ".pdf", bbox_inches='tight')
    plt.close(fig)
    return fig


# %%
plot_avg_cumulatives(
    conf=0.99,
    save_fig=False,
)
# %%


def plot_hardness_full(
    NQ: int = 44,
    N_runs: int = 2000,
    style: str = "lin",
    save_fig: bool = True
) -> None:
    '''
    Returns a figure for the cumulative passes of E vs N

    Return:
        fig (matplotlib figure): FIgure with all the... TODO

    Arguments:
        NQ (int): Number of qubits
        N_runs (int): Number of runs for plot
        style = "lin" (str): String of instances to be plotted
        save_fig = True (bool): Boolean to save figure
    '''

    df_total_file_avg = "../code_for_art/normalized_prob_dist/QPSK/avg/all_avg/avg_pass_test_N=" + \
        str(N_runs)+".csv"
    df_total_file_opt = "../code_for_art/normalized_prob_dist/QPSK/opt/all_avg/avg_pass_test_N=" + \
        str(N_runs)+".csv"
    # Resource usage for each run (37 constant + 0.127 factor)
    N_runs_list = np.array(range(1, N_runs + 1)) * 0.127 + 37

    df_total_avg = pd.read_csv(df_total_file_avg, index_col=0)
    df_total_opt = pd.read_csv(df_total_file_opt, index_col=0)

    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((2, 5), (0, 2),  rowspan=2)  # , sharey=ax1)
    ax3 = plt.subplot2grid((2, 5), (0, 3),  rowspan=2)  # , sharey=ax2)
    ax4 = plt.subplot2grid((2, 5), (0, 4),  rowspan=2)  # , sharey=ax2)
#     ax5 = plt.subplot2grid((2,6),(0,5),  rowspan=2, sharey=ax1)

    fig.subplots_adjust(wspace=0)
    # print("generation of grid plots - OK")
    # fig.set_figheight(16)
    # fig.set_figwidth(24)
    # f_size = 40
    thickness = 1

    # ax1.tick_params(axis='y',labelsize = f_size)
    # ax1.tick_params(axis='x',labelsize = f_size)

    # ax2.tick_params(axis='y',labelsize = f_size)
    # ax2.tick_params(axis='x',labelsize = f_size)

    # ax3.tick_params(axis='y',labelsize = f_size)
    # ax3.tick_params(axis='x',labelsize = f_size)

    # ax4.tick_params(axis='y',labelsize = f_size)
    # ax4.tick_params(axis='x',labelsize = f_size)
#     ax1.set_ylabel("E", color="black", fontsize = f_size)
#     print(len(df_total_avg["avg"]), len(N_runs_list))
#     print(len(df_total_opt["avg"]))
    label_avg = "Fixed"
    label_opt = "Optimal w/o OH"
    label_oh = "Optimal w/ OH"
    # time to find optimal parameter setting
    opt_find_time = 200

    if style == "log":
        ax1.plot(np.log10(N_runs_list),
                 df_total_avg["avg"], c="r", label=label_avg, linewidth=thickness)
        ax1.plot(np.log10(N_runs_list), df_total_opt["avg"], "--", c="xkcd:hot pink",
                 label=label_opt, linewidth=thickness, markevery=400, markersize=12)
        ax1.plot(np.log10(N_runs_list[:-1000] + opt_find_time), df_total_opt["avg"].values[:-
                 1000], c="xkcd:hot pink", label=label_oh, linewidth=thickness)
        xlim = ax1.get_ylim()
        new_xlim = (np.log10(37), np.log10(N_runs))
        ax1.set_xlim(new_xlim)
        ax1xticks = [1.5, 2, 3]
        ticks_dict = {1.5: "", 2: r"$10^2$", 3: r"$10^3$"}
        ax1.set_xticks(ax1xticks)
        labels = [ax1xticks[i] if t not in ticks_dict.keys() else ticks_dict[t]
                  for i, t in enumerate(ax1xticks)]
        ax1.set_xticklabels(labels)
        ax1.arrow(np.log10(opt_find_time * 0.127 + 37), df_total_opt["avg"].values[opt_find_time], np.log10(opt_find_time * 0.127 + 37+opt_find_time)-np.log10(
            opt_find_time * 0.127 + 37), 0,  head_width=0.01, head_length=0.05, linewidth=4, color="black", length_includes_head=True)
        ax1.arrow(np.log10(opt_find_time * 0.127 + 37+opt_find_time), df_total_opt["avg"].values[opt_find_time], -(np.log10(opt_find_time * 0.127 + 37+opt_find_time)-np.log10(opt_find_time * 0.127 + 37)), 0,  head_width=0.01, head_length=0.05, linewidth=4, color="black", length_includes_head=True,
                  zorder=10)
        ax1.annotate("Overhead", xy=(np.log10(280 * 0.127 + 37),
                     df_total_opt["avg"].values[opt_find_time] + 0.005))  # , fontsize=f_size-10)

    elif style == "lin":
        ax1.plot(N_runs_list, df_total_avg["avg"],
                 c="r", label=label_avg, linewidth=thickness)
        ax1.plot(N_runs_list, df_total_opt["avg"],
                 c="xkcd:magenta", label=label_opt, linewidth=thickness)
        ax1.plot(N_runs_list[:-1000] + 100, df_total_opt["avg"].values[:-
                 1000], c="xkcd:purple", label=label_oh, linewidth=thickness)

        xlim = ax1.get_ylim()
        new_xlim = (25, 1350)
        ax1.set_xlim(new_xlim)

    ylim = ax1.get_ylim()
    new_ylim = (-0.01, 0.32)
    ax1.set_ylim(new_ylim)

#     xlim = ax1.get_ylim()
#     new_xlim = (1, 4)
#     ax1.set_xlim(new_xlim)

    ax1.set_xlabel(r'$R = t\ [ms]$')  # , fontsize = f_size)
    ax1.set_ylabel("E")  # ,  fontsize = f_size)
#     fig.legend(loc=(0.9,0.8))
#     plt.show()

    path10 = "../code_for_art/normalized_prob_dist/QPSK/avg/N=" + str(10) + "/"
    path100 = "../code_for_art/normalized_prob_dist/QPSK/avg/N=" + \
        str(100) + "/"
    path1000 = "../code_for_art/normalized_prob_dist/QPSK/avg/N=" + \
        str(1000) + "/"

    avg_cumulative10 = np.zeros(1001)
    avg_cumulative100 = np.zeros(1001)
    avg_cumulative1000 = np.zeros(1001)
    opacity = 0.3
    for problem in range(1, 21):
        df_cum10 = pd.read_csv(path10 + "cumulative_norm_NQ=" + str(44) +
                               "_problem=" + str(problem) + "_N=" + str(10) + ".csv", index_col=0)
        ax2.plot(df_cum10["eng"], df_cum10["cumulative"],
                 c="grey", alpha=opacity, linewidth=thickness-2)
        avg_cumulative10 += df_cum10["cumulative"].values/20

        df_cum100 = pd.read_csv(path100 + "cumulative_norm_NQ=" + str(44) +
                                "_problem=" + str(problem) + "_N=" + str(100) + ".csv", index_col=0)
        ax3.plot(df_cum100["eng"], df_cum100["cumulative"],
                 c="grey", alpha=opacity, linewidth=thickness-2)
        avg_cumulative100 += df_cum100["cumulative"].values/20

        df_cum1000 = pd.read_csv(path1000 + "cumulative_norm_NQ=" + str(
            44) + "_problem=" + str(problem) + "_N=" + str(1000) + ".csv", index_col=0)
        ax4.plot(df_cum1000["eng"], df_cum1000["cumulative"],
                 c="grey", alpha=opacity, linewidth=thickness-2)
        avg_cumulative1000 += df_cum1000["cumulative"].values/20

    # , markersize = 12, label = "Mean")
    ax2.plot(df_cum10["eng"], avg_cumulative10,
             c="orange", linewidth=thickness)
    ax3.plot(df_cum100["eng"], avg_cumulative100,  c="b",
             linewidth=thickness)  # , label = "Mean")
    ax4.plot(df_cum1000["eng"], avg_cumulative1000, c="g",
             linewidth=thickness)  # , label = "Mean")
    ax2.set_title("#N=10")  # , fontsize=f_size)
    ax3.set_title("#N=100")  # , fontsize=f_size)
    ax4.set_title("#N=1000")  # , fontsize=f_size)

    for idx, conf in enumerate(avg_cumulative10):
        if conf >= 0.9:
            E_pass_test10 = df_cum10["eng"][idx]
            break

    for idx, conf in enumerate(avg_cumulative100):
        if conf >= 0.9:
            E_pass_test100 = df_cum100["eng"][idx]
            break

    for idx, conf in enumerate(avg_cumulative1000):
        if conf >= 0.9:
            E_pass_test1000 = df_cum1000["eng"][idx]
            break

    ax2.axhline(0.90, xmin=0, xmax=1, c="navy",
                ls=":", linewidth=3.5, zorder=0)
    ax2.axvline(E_pass_test10, c="navy", ls=":", linewidth=3.5, zorder=0)

    ax3.axhline(0.90, xmin=0, xmax=1, c="navy",
                ls=":", linewidth=3.5, zorder=0)
    ax3.axvline(E_pass_test100, c="navy", ls=":", linewidth=3.5, zorder=0)

    ax4.axhline(0.90, xmin=0, xmax=1, c="navy",
                ls=":", linewidth=3.5, zorder=0)
    ax4.axvline(E_pass_test1000, c="navy", ls=":", linewidth=3.5, zorder=0)

    ylim = ax2.get_ylim()
    new_ylim = (-0.01, 1.01)
    ax2.set_ylim(new_ylim)

    labels = [item.get_text() for item in ax2.get_yticklabels()]

    empty_string_labels = ['']*len(labels)
    ax2.set_yticklabels(empty_string_labels)

    ylim = ax3.get_ylim()
    ax3.set_ylim(new_ylim)

    ylim = ax2.get_ylim()
    ax4.set_ylim(new_ylim)

#         ylim = ax2.get_ylim()
#     new_ylim = (-0.01, 1.0)
#     ax2.set_ylim(new_ylim)

    xlim = ax2.get_xlim()
    new_xlim = (-0.01, 0.35)
    ax2.set_xlim(new_xlim)

    xlim = ax3.get_xlim()
    ax3.set_xlim(new_xlim)
    ax3.set_yticklabels(empty_string_labels)

    xlim = ax4.get_xlim()
    ax4.set_xlim(new_xlim)

    ax4.yaxis.tick_right()

    ax3.set_xlabel('E')  # , fontsize = f_size)
    ax4.set_ylabel("Probability")  # ,  fontsize = f_size)
    ax4.yaxis.set_label_position("right")
#     ax.set_xlabel('E', fontsize = 25)
#     ax.set_ylabel("Probability",  fontsize = 25)
    fig.legend(loc=(0.17, 0.77), framealpha=1)  # , prop = {"size" : f_size-5})

    if save_fig:

        path_plots = "new_plots/hardness/"
        if not os.path.exists(path_plots):
            os.makedirs(path_plots)

        title = "hardnes_NQ=" + str(NQ) + ".pdf"

#     fig.savefig(path_plots + title +  ".pdf", bbox_inches='tight')
#     plt.close(fig)

        path_plots = "../code_for_art/new_plots/hardness_total/"
        if not os.path.exists(path_plots):
            os.makedirs(path_plots)

        title = "hardness_total_" + style

        fig.savefig(path_plots + title + ".pdf", bbox_inches='tight')
    plt.close(fig)
    return fig


# %%
plot_hardness_full(
    N_runs=10000,
    style='log',
    save_fig=False
)
# %%
