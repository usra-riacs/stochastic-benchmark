# TODO This code needs to be functionalized for it to be imported into data analysis code
# %%
# Create all instances and save it into disk
for instance in instance_list:
    instance_file_name = prefix + str(instance) + ".txt"
    instance_file_name = os.path.join(instance_path, instance_file_name)

    if not os.path.exists(instance_file_name):
        # Fixing the random seed to get the same result
        np.random.seed(instance)
        J = np.random.rand(N, N)
        # We only consider upper triangular matrix ignoring the diagonal
        J = np.triu(J, 1)
        h = np.random.rand(N)
        model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)

        text_file = open(instance_file_name, "w")
        text_file.write(model_random.to_coo())
        text_file.close()
# %%
# Run all instances with PySA
# Parameters: replicas = [1, 2, 4, 8], n_reads = 100, sweeps = 100, p_hot=50, p_cold = 1
# Input Parameters
total_reads = 1000
n_replicas_list = [1, 2, 4, 8]
# n_replicas_list = [4]
# sweeps = [i for i in range(
#     1, 21, 1)] + [i for i in range(
#         21, 101, 10)]
p_hot_list = [50.0]
p_cold_list = [1.0]
# instance_list = list(range(20)) + [42]
# instance_list = [1, 4, 11, 14, 15, 16] + [42]
# instance_list = [0,2,3,5,6,7,8,9,10,12,13,17,18,19]
# instance_list = [42]
use_raw_pickles = True
overwrite_pickle = False
float_type = 'float32'

# sweeps_list = [i for i in range(1, 21, 1)] + [
#     i for i in range(20, 501, 10)] + [
#     i for i in range(500, 1001, 20)]
# sweeps = [1000]

# Setup directory for PySA results
pysa_path = os.path.join(results_path, "pysa/")
# Create directory for PySA results
if not os.path.exists(pysa_path):
    os.makedirs(pysa_path)

if use_raw_pickles:

    # Setup directory for PySA pickles
    pysa_pickles_path = os.path.join(pysa_path, "pickles/")
    # Create directory for PySA pickles
    if not os.path.exists(pysa_pickles_path):
        os.makedirs(pysa_pickles_path)

    # List all the instances files
    file_list = createInstanceFileList(directory=instance_path,
                                       instance_list=instance_list)

    counter = 0
    for file in file_list:

        file_name = file.split(".txt")[0].rsplit("/", 1)[-1]
        print(file_name)

        data = np.loadtxt(file, dtype=float)
        M = sparse.coo_matrix(
            (data[:, 2], (data[:, 0], data[:, 1])), shape=(N, N))
        problem = M.A
        problem = problem+problem.T-np.diag(np.diag(problem))

        # Get solver
        solver = Solver(problem=problem, problem_type='ising',
                        float_type=float_type)

        for n_replicas in n_replicas_list:
            for n_sweeps in sweeps_list:
                for p_hot in p_hot_list:
                    for p_cold in p_cold_list:
                        counter += 1
                        print("file "+str(counter)+" of " + str(len(file_list)
                                                                * len(n_replicas_list) * len(sweeps_list) * len(p_hot_list) * len(p_cold_list)))

                        pickle_name = pysa_pickles_path + file_name + '_swe_' + str(n_sweeps) + '_rep_' + str(
                            n_replicas) + '_pcold_' + str(p_cold) + '_phot_' + str(p_hot) + '.pkl'

                        min_temp = 2 * \
                            np.min(np.abs(problem[np.nonzero(problem)])
                                   ) / np.log(100/p_cold)
                        min_temp_cal = 2*min(sum(abs(i)
                                                 for i in problem)) / np.log(100/p_cold)
                        max_temp = 2*max(sum(abs(i)
                                             for i in problem)) / np.log(100/p_hot)
                        if os.path.exists(pickle_name) and not overwrite_pickle:
                            print(pickle_name)
                            results_pysa = pd.read_pickle(pickle_name)
                            continue
                        print(pickle_name)
                        # Apply Metropolis
                        results_pysa = solver.metropolis_update(
                            num_sweeps=n_sweeps,
                            num_reads=total_reads,
                            num_replicas=n_replicas,
                            update_strategy='random',
                            min_temp=min_temp,
                            max_temp=max_temp,
                            initialize_strategy='random',
                            recompute_energy=True,
                            sort_output_temps=True,
                            parallel=True,  # True by default
                            use_pt=True,
                            verbose=False,
                        )
                        results_pysa.to_pickle(pickle_name)

# %%
# Compute preliminary ground state file with best found solution by PySA
compute_pysa_gs = True

if compute_pysa_gs:
    for instance in instance_list:
        # List all the pickled filed for an instance files
        pickle_list = createPySAExperimentFileList(
            directory=pysa_pickles_path,
            instance_list=[instance],
            prefix=prefix,
        )
        min_energies = []
        for file in pickle_list:
            df = pd.read_pickle(file)
            min_energies.append(df['best_energy'].min())

        with open(os.path.join(results_path, "gs_energies.txt"), "a") as gs_file:
            gs_file.write(prefix + str(instance) + " " +
                          str(np.nanmin(min_energies)) + " " + "best_found pysa\n")


# %%
# Load minimum found energy across each instance
def getMinPySAEnergy(
    directory: str,
    instance: Union[str, int],
    prefix: str = "",
) -> float:
    '''
    Load minimum found energy across each instance

    Args:
        directory: Directory where the PySA pickles are located
        instance: Instance number
        prefix: Prefix of the instance file

    Returns:
        Minimum found energy
    '''
    # instance = int(instance_name.rsplit("_",1)[1])
    min_energies = [
        df_dneal[df_dneal['instance'] == instance]['best'].min()]
    file_list = createPySAExperimentFileList(
        directory=directory,
        instance_list=[instance],
        prefix=prefix,
    )
    for file in file_list:
        df = pd.read_pickle(file)
        min_energies.append(df['best_energy'].min())
    return np.nanmin(min_energies)


# %%
# Create intermediate .data files with main information and unique_gs with unique groundstates information

# Set up directory for intermediate .data files
pysa_data_path = os.path.join(pysa_path, "data/")
# Create directory for intermediate .data files
if not os.path.exists(pysa_data_path):
    os.makedirs(pysa_data_path)


# Setup directory for unique ground states
pysa_gs_path = os.path.join(pysa_path, "unique_gs/")
# Create directory for unique ground states
if not os.path.exists(pysa_gs_path):
    os.makedirs(pysa_gs_path)

# Percentual tolerance to consider succesful runs
tol = 1

if use_raw_pickles:
    overwrite_files = True
    output_files_in_progress = []

    counter = 0
    for instance in instance_list:

        min_energy = loadEnergyFromFile(gs_file=results_path + "gs_energies.txt",
                                        instance_name=prefix + str(instance))

        # List all the instances files
        pickle_list = createPySAExperimentFileList(
            directory=pysa_pickles_path,
            instance_list=[instance],
            rep_list=n_replicas_list,
            sweep_list=sweeps_list,
            pcold_list=p_cold_list,
            phot_list=p_hot_list,
            prefix=prefix,
        )
        # print(pickle_list)
        for pickle_file in pickle_list:
            file_name = pickle_file.split(".pkl")[0].rsplit("/", 1)[-1]
            instance_name = file_name.split("_swe")[0]
            counter += 1
            output_file = pysa_data_path+file_name+".data"
            if overwrite_files or not os.path.exists(output_file):
                print(file_name + ": file "+str(counter) +
                      " of "+str(len(pickle_list)*len(instance_list)))
                if os.path.exists(pickle_file):
                    try:
                        results_pysa = pd.read_pickle(pickle_file)
                    except (pkl.UnpicklingError, EOFError):
                        os.replace(pickle_file, pickle_file + '.bak')
                        continue
                else:
                    print("Missing pickle file for " + file_name)
                    break

                sweeps_max = getSweepsPySAExperiment(pickle_file)

                num_sweeps = results_pysa["num_sweeps"][0]

                # Check that each file has as many reads as required
                n_reads_file = len(results_pysa["best_energy"])
                assert(total_reads == n_reads_file)

                if os.path.exists(output_file) and not(output_file in output_files_in_progress):
                    output_files_in_progress.append(output_file)
                    with open(output_file, "w") as fout:
                        fout.write("s read_num runtime(us) num_sweeps success_e" +
                                   str(-int(np.log10(tol)))+"\n")

                states_within_tolerance = []
                # Skip first read, as numba needs compile time
                runtimes = []
                successes = []
                for read_num in range(1, n_reads_file):
                    best_energy = results_pysa["best_energy"][read_num]
                    runtime = results_pysa["runtime (us)"][read_num]
                    runtimes.append(runtime)

                    best_sweeps = num_sweeps
                    success = 0
                    if(abs(float(best_energy)-float(min_energy))/float(min_energy) < tol/100):
                        best_sweeps = results_pysa["min_sweeps"][read_num]
                        states_within_tolerance.append(
                            results_pysa["best_state"][read_num])
                        success = 1
                    successes.append(success)

                    with open(output_file, "a") as fout:
                        fout.write("{s} {read} {runtime} {best_sweeps} {success}\n".
                                   format(s=num_sweeps,
                                          read=read_num,
                                          runtime=runtime,
                                          best_sweeps=best_sweeps,
                                          success=success))

                # Separate file with unique MAXCUT per instance
                unique_gs = np.unique(np.asarray(
                    states_within_tolerance), axis=0)
                with open(pysa_gs_path+output_file.split("/")[-1], "a") as fout:
                    fout.write("tol{tol} s{s} unqGS{unqGS} \n".format(
                        tol=-int(np.log10(tol)), s=num_sweeps, unqGS=len(unique_gs)))
            else:
                data = np.loadtxt(output_file, skiprows=1,
                                  usecols=(0, 1, 2, 3, 4))
                successes = data[:, 4]
                best_sweeps = data[:, 3]
                runtimes = data[:, 2]
# %%
# Create pickled Pandas framework with results for each instance
for instance in instance_list:
    data_dict_name = "results_" + str(instance) + "T.pickle"
    df_name = "df_results_" + str(instance) + "T.pickle"

    file_list = createPySAExperimentFileList(
        directory=pysa_data_path,
        instance_list=[instance],
        rep_list=n_replicas_list,
        sweep_list=sweeps_list,
        pcold_list=p_cold_list,
        phot_list=p_hot_list,
        prefix=prefix,
    )

    data_dict_path = os.path.join(pysa_path, data_dict_name)
    df_path = os.path.join(pysa_path, df_name)
    data_dict = {}
    counter = 0

    tts_list = []
    tts_scaled_list = []
    for file in file_list:
        counter += 1
        file_name = file.split(".data")[0].rsplit(
            "/", 1)[-1]
        print(file_name + ": file "+str(counter)+" of "+str(len(file_list)))

        # If you wanto to use the raw data and process it here
        if use_raw_data or not(os.path.exists(data_dict_path)) or not(os.path.exists(df_path)):
            instance = getInstancePySAExperiment(file)
            sweep = getSweepsPySAExperiment(file)
            replica = getReplicas(file)
            pcold = getPCold(file)
            phot = getPHot(file)
            # load in data, parameters
            # s,sweeps,runtime(us),best_sweeps,success
            data = np.loadtxt(file, skiprows=1, usecols=(0, 1, 2, 3, 4))

            # Computation of TTS across mean value of all reads in each PySA run
            successes = data[:, 4]
            best_sweeps = data[:, 3]
            runtimes = data[:, 2]
            success_rate = np.mean(successes)
            mean_time = np.mean(runtimes)  # us
            if success_rate == 0:
                tts_scaled = 1e15
                tts = 1e15
            # Consider continuous TTS and TTS scaled by assuming s=1 as s=1-1/1000*(1-1/10)
            elif success_rate == 1:
                tts_scaled = 1e-6*mean_time * \
                    np.log(1-s) / np.log(1-0.999+0.0001)  # s
                tts = replica*1e-6*mean_time * \
                    np.log(1-s) / np.log(1-0.999+0.00001)  # s * replica
            else:
                tts_scaled = 1e-6*mean_time * \
                    np.log(1-s) / np.log(1-success_rate)  # s
                tts = replica*1e-6*mean_time * \
                    np.log(1-s) / np.log(1-success_rate)  # s * replica
            tts_scaled_list.append(tts_scaled)
            tts_list.append(tts)
            if instance not in data_dict.keys():
                data_dict[instance] = {}
            if sweep not in data_dict[instance].keys():
                data_dict[instance][sweep] = {}
            if replica not in data_dict[instance][sweep].keys():
                data_dict[instance][sweep][replica] = {}
            if pcold not in data_dict[instance][sweep][replica].keys():
                data_dict[instance][sweep][replica][pcold] = {}
            if phot not in data_dict[instance][sweep][replica][pcold].keys():
                data_dict[instance][sweep][replica][pcold][phot] = {}

            # data_dict[instance][sweep][replica][pcold][phot]['success'] = data[:,4]
            data_dict[instance][sweep][replica][pcold][phot]['best_sweep'] = best_sweeps
            data_dict[instance][sweep][replica][pcold][phot]['success_rate'] = success_rate
            data_dict[instance][sweep][replica][pcold][phot]['mean_time'] = mean_time
            data_dict[instance][sweep][replica][pcold][phot]['tts'] = tts
            data_dict[instance][sweep][replica][pcold][phot]['tts_scaled'] = tts_scaled
            if(len(data[:, 1]) < replica):
                print('Missing replicas for instance' + str(file))
                print(len(data[:, 1]))
                pass

            # Save results dictionary in case that we are interested in reusing them
            pkl.dump(data_dict, open(data_dict_name, "wb"))

            # Create Pandas framework for the results
            # Restructure dictionary to dictionary of tuple keys -> values
            data_dict_2 = {(instance, sweep, replica, pcold, phot):
                           (data_dict[instance][sweep][replica][pcold][phot]['best_sweep'],
                           data_dict[instance][sweep][replica][pcold][phot]['success_rate'],
                           data_dict[instance][sweep][replica][pcold][phot]['mean_time'],
                           data_dict[instance][sweep][replica][pcold][phot]['tts'],
                           data_dict[instance][sweep][replica][pcold][phot]['tts_scaled'])
                           for instance in data_dict.keys()
                           for sweep in data_dict[instance].keys()
                           for replica in data_dict[instance][sweep].keys()
                           for pcold in data_dict[instance][sweep][replica].keys()
                           for phot in data_dict[instance][sweep][replica][pcold].keys()}

            # Construct dataframe from dictionary
            df_dneal = pd.DataFrame.from_dict(
                data_dict_2, orient='index').reset_index()

            # Split column of tuples to multiple columns
            df_dneal[['instance', 'sweeps', 'replicas', 'pcold',
                      'phot']] = df_dneal['index'].apply(pd.Series)

            # Clean up: remove unwanted columns, rename and sort
            df_dneal = df_dneal.drop('index', 1).\
                rename(columns={0: 'best_sweep', 1: 'success_rate', 2: 'mean_time', 3: 'tts', 4: 'tts_scaled'}).\
                sort_index(axis=1)

            df_dneal.to_pickle(df_path)
        else:  # Just reload processed datafile
            # data_dict = pkl.load(open(results_name, "rb"))
            pass
# %%