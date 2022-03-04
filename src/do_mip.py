# TODO This code needs to be functionalized for use in other scripts.
# %%
# Compute minimum value using MIP
# Ground state computation
compute_mip_gs = False
# Which type of MIP formulation to use ("qubo", "lcbo", "qcbo")
mip_formulation = "qubo"
if not compute_mip_gs:
    pass
else:
    import pyomo.environ as pyo
    from pyomo.opt import SolverStatus, TerminationCondition

    # Obtain ground states if possible using Mixed-Integer Formulation through Pyomo
    # Set up MIP optimization results directory
    mip_results_path = os.path.join(results_path, "mip_results/")
    # Create unexisting directory
    if not os.path.exists(mip_results_path):
        print('MIP results ' + mip_results_path +
              ' does not exist. We will create it.')
        os.makedirs(mip_results_path)
    # Compute optimal solution using MIP and save it into ground state file
    # Other solvers are available using GLPK, CBC (timeout), GAMS, Gurobi, or CPLEX
    solver_name = "gurobi"
    mip_solver = pyo.SolverFactory(solver_name)
    bqm_bin = model_random.change_vartype("BINARY", inplace=False)
    offset = bqm_bin.offset
    nx_graph_bin = bqm_bin.to_networkx_graph()

    # Create instance
    pyo_model = pyo.ConcreteModel(name="Random SK problem " + str(instance))
    # Define variables
    # Node variables
    pyo_model.x = pyo.Var(nx_graph_bin.nodes(), domain=pyo.Binary)
    obj_expr = offset
    for i, val in nx.get_node_attributes(nx_graph_bin, 'bias').items():
        obj_expr += val * pyo_model.x[i]

    if mip_formulation == "qubo":
        # Direct QUBO formulation
        obj_expr += pyo.quicksum(nx_graph_bin[i][j]['bias'] * pyo_model.x[i] * pyo_model.x[j]
                                 for (i, j) in nx_graph_bin.edges())
        # for (i, j) in nx_graph_bin.edges():
        #     # We want all edges to be sorted  with i-j and i<j
        #     assert(i < j)
        #     if i != j:
        #         obj_expr += nx_graph_bin[i][j]['bias'] * instance.x[i]*instance.x[j]
        #     else:
        #         print("Graph with self-edges" + str(i))

    elif mip_formulation == "lcbo":
        # Linear Constrained Binary Optimization

        # Edge variables
        pyo_model.y = pyo.Var(nx_graph_bin.edges(), domain=pyo.Binary)

        # add model constraints
        pyo_model.c1 = pyo.ConstraintList()
        pyo_model.c2 = pyo.ConstraintList()
        pyo_model.c3 = pyo.ConstraintList()

        for (i, j) in nx_graph_bin.edges():
            # We want all edges to be sorted  with i-j and i<j
            assert(i < j)
            if i != j:
                pyo_model.c1.add(pyo_model.y[i, j] <=
                                 pyo_model.x[i])
                pyo_model.c2.add(pyo_model.y[i, j] <=
                                 pyo_model.x[j])
                pyo_model.c3.add(pyo_model.y[i, j] >=
                                 pyo_model.x[i] + pyo_model.x[j] - 1)
                obj_expr += nx_graph_bin[i][j]['bias'] * pyo_model.y[i, j]
            else:
                print("Graph with self-edges" + str(i))
    else:
        print("Formulation not implemented yet")

    # Define the objective function
    pyo_model.objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)
    # Solve
    if solver_name == "gurobi":
        mip_solver.options['NonConvex'] = 2
        mip_solver.options['MIPGap'] = 1e-9
        mip_solver.options['TimeLimit'] = 30
    elif solver_name == "gams":
        mip_solver.options['solver'] = 'baron'
        mip_solver.options['solver'] = 'baron'
        mip_solver.options['add_options'] = 'option reslim=10;'

    results_dneal = mip_solver.solve(
        pyo_model,
        tee=True,
    )
    # result = opt_gams.solve(instance, tee=True)
    # Save solution
    obj_val = pyo_model.objective.expr()
    opt_sol = pd.DataFrame.from_dict(
        pyo_model.x.extract_values(), orient="index", columns=[str(pyo_model.x)])
    # Missing transformation back to spin variables here

    if (results_dneal.solver.status == SolverStatus.ok) and (results_dneal.solver.termination_condition == TerminationCondition.optimal):
        opt_sol.to_csv(mip_results_path + instance_name + "_" + str(obj_val) +
                       "_opt_sol.txt", header=None, index=True, sep=" ")
        with open(mip_results_path + "gs_energies.txt", "a") as gs_file:
            gs_file.write(instance_name + " " +
                          str(obj_val) + " " + str(results_dneal.solver.time) + " " + mip_formulation + " " + solver_name + "\n")

    else:
        opt_sol.to_csv(mip_results_path + instance_name + "_" + str(obj_val) +
                       "_sol.txt", header=None, index=True, sep=" ")
        with open(mip_results_path + "gs_energies.txt", "a") as gs_file:
            gs_file.write(instance_name + " " +
                          str(obj_val) + " " + str(results_dneal.solver.time) +
                          " " + str(results_dneal.solver.gap) + " " + mip_formulation + " " + solver_name + " suboptimal\n")
