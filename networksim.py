import pathlib
import numpy as np
import datetime
import matplotlib.pyplot as plt
import _pickle
import theoretical_calculations
import bitcoinsimulationD
import TheorySimulationBitcoin
import simulationEth
import TheorySimulationEthereum
import randomNet
import gc

########################################################################################################################
# configurations
########################################################################################################################
from main import compare_strategies

N = 3000  # nodes
runs = 0  # take average from these runs, if zero: loads from specified path
do_full_sim = True
do_theory_and_lite = 0
pathlength = 4  # theoretical paths are computed until this length
steps = 10  # how many nodes should be left out in theoretical shortest paths
polate = 0  # which degree should be used for interpolations, 0 -> no interpolation
properties = {"deg": 0, "clu": 1, "pat": 0, "clo": 0}  # choose which properties you want calculated
plot_path = './data/comparisons/'
db_path = './data/objects/'
sim_load_from = './data/objects/2021_3_8/14_33_36_3000_'  # specify which data to load
db_load_from = db_path + "2021_3_13/10_37_46_3000_DB.pkl"

########################################################################################################################
# initialization
########################################################################################################################
theo_strats = []
sim_strats = []
deg_clu_strats = []
pat_clo_strats = []
if do_full_sim:
    sim_strats += ["full sim bitcoin", "full sim ethereum"]
    deg_clu_strats += [["full sim bitcoin", "full sim ethereum"]]
    pat_clo_strats += [["full sim bitcoin", "full sim ethereum"]]
if do_theory_and_lite:
    theo_strats += ["theoretical bitcoin", "theoretical ethereum"]
    sim_strats += ["random bitcoin", "random ethereum", "lite sim bitcoin", "lite sim ethereum"]
    deg_clu_strats += [["theoretical bitcoin", "theoretical ethereum", "lite sim bitcoin", "lite sim ethereum"]]
    pat_clo_strats += [["random bitcoin", "lite sim bitcoin", "random ethereum", "lite sim ethereum"]]
DB = dict()  # stores final data for all strats
for strat in sim_strats + theo_strats:  # initializes DB
    DB[strat] = dict()
    for s in properties.keys():
        DB[strat][s] = dict()


ti = datetime.datetime.now()
s = str(ti.year) + '_' + str(ti.month) + '_' + str(ti.day) + '/' + str(ti.hour) + '_' +\
       str(ti.minute) + '_' + str(ti.second) + '_' + str(N)
plot_path += s
db_path += s
p1 = pathlib.Path(plot_path)
p2 = pathlib.Path(db_path)
if not p1.exists():
    p1.mkdir(parents=True, exist_ok=True)
if not p2.exists():
    p2.mkdir(parents=True, exist_ok=True)

########################################################################################################################
# main
########################################################################################################################

def main():
    t_start = datetime.datetime.now()

    if runs == 0:
        load_data()
    else:
        run_simulations()
    plot_strategies()
    if runs > 0:
        store_DB()

    t_end = datetime.datetime.now()
    print("finished everything, total time elapsed: ", (t_end - t_start).total_seconds() / 60, " minutes. ")

########################################################################################################################
# main functions
########################################################################################################################

def run_simulations():
    """
    calculates theory once and runs the simulations specified in sim_strats for #runs specified in runs
    and stores every simulation instance in sims
    """
    global DB
    #run theoretical once
    storage = './data/objects/2021_3_7/16_9_27_3000_'  # specify which data to load
    for strat in theo_strats:
        with open(storage + strat + '.pkl', 'rb') as input:
            sim = _pickle.load(input)
            take_average_of_runs([compute_results(strat, sim)], strat)
        #sims[strat][1] = do_theoretical_calculations(strat, properties)

    #run all sims "runs" times
    for strat in sim_strats:
        runsss = list()
        for run in range(1, runs+1):
            print("run ", run, " of ", runs, " in ", strat)
            sim = do_simulation(strat, properties)
            runsss.append(compute_results(strat, sim))
        if runs > 1:
            runsss = cleanup(runsss)
        take_average_of_runs(runsss, strat)
        del runsss
        print("gc at work")
        gc.collect()
        print("gc done")


def compute_results(strat, sim):
    """
    gets results of properties and stores theo in average_DB and sims in in DB,
    if runs = 1: copies sims to average_DB
    """
    global properties

    temp_result = dict()  # used temporarily to store the results of a single run and strat
    for s in properties.keys():
        temp_result[s] = dict()

    # theory calculations
    if strat in theo_strats:
        temp_result["deg"]["dat"], temp_result["deg"]["avg"], temp_result["deg"]["var"], \
            temp_result["deg"]["std"] = sim.get_degree_results(plotsolo=False, polate=polate)
        temp_result["clu"]["dat"], temp_result["clu"]["avg"], temp_result["clu"]["var"], \
            temp_result["clu"]["std"] = sim.get_clustering_results(plotsolo=False, polate=polate)
    # sim results
    else:
        temp_result["deg"]["dat"], temp_result["deg"]["avg"], temp_result["deg"]["var"], temp_result["deg"]["std"] = sim.whiteboard.plot_degree_in_time(plotsolo=False, polate=polate)
        temp_result["clu"]["dat"], temp_result["clu"]["avg"], temp_result["clu"]["var"], temp_result["clu"]["std"] = sim.whiteboard.clustering_in_time(plotsolo=False, polate=polate)
        temp_result["pat"]["hops"], temp_result["pat"]["dat"], temp_result["pat"]["avg"] = sim.whiteboard.shortest_path_histogram_undirected(plotsolo=False, is_final=False)
        temp_result["clo"]["dat"], temp_result["clo"]["avg"], temp_result["clo"]["var"], temp_result["clo"]["std"] = sim.whiteboard.closeness_in_time(plotsolo=False, polate=polate)

    return temp_result


def cleanup(runsss):
    """
    :param runsss: list of results from multiple runs
    :return: cleaned up (all same length) results for averaging
    """
    minn = N
    maxp = 0
    for run in runsss:
        minn = min(minn, len(run["deg"]["dat"]))
        maxp = max(maxp, len(run["pat"]["dat"]))
    for run in runsss:
        run["deg"]["dat"] = run["deg"]["dat"][:minn]
        run["clu"]["dat"] = run["clu"]["dat"][:minn]
        run["clo"]["dat"] = run["clo"]["dat"][:minn]
        while len(run["pat"]["dat"]) < maxp:
            run["pat"]["dat"].append(0)
    return runsss


def take_average_of_runs(runs, strat):
    """
    takes average of all runs and stores it in DB
    :param runs: list of results of multiple runs, strat is the strategy
    :return: nothing, stores in global DB
    """
    global DB
    if len(runs) > 1:
        l = len(runs[0]["deg"]["dat"])
        p = len(runs[0]["pat"]["dat"])
        DB[strat]["deg"]["dat"] = np.zeros(l)
        DB[strat]["clu"]["dat"] = np.zeros(l)
        DB[strat]["pat"]["dat"] = np.zeros(p)
        DB[strat]["clo"]["dat"] = np.zeros(l)
        for run in runs:
            r = len(runs)
            DB[strat]["deg"]["dat"] = np.add(DB[strat]["deg"]["dat"], (np.array(run["deg"]["dat"]) / r))
            DB[strat]["clu"]["dat"] = np.add(DB[strat]["clu"]["dat"], (np.array(run["clu"]["dat"]) / r))
            DB[strat]["pat"]["dat"] = np.add(DB[strat]["pat"]["dat"], (np.array(run["pat"]["dat"]) / r))
            DB[strat]["clo"]["dat"] = np.add(DB[strat]["clo"]["dat"], (np.array(run["clo"]["dat"]) / r))
    else:
        DB[strat] = runs[0]


def store_DB():
    global DB

    with open(db_path +'_DB.pkl', 'wb') as output:
        _pickle.dump(DB, output)


def plot_strategies():
    """
    plots data of average_DB
    todo: only take N of theo results
    """
    global DB

    dot = "."

    # degree
    if properties["deg"]:
        for ll in deg_clu_strats:
            plt.title('Degree')
            plt.ylabel('degree')
            plt.xlabel("node id")
            for strat in ll:  # strats
                #print(strat, DB[strat])
                #print(DB[strat][0]["deg"])
                #print(DB[strat][0]["deg"]["dat"])
                y = DB[strat]["deg"]["dat"]
                x = range(len(y))
                if len(ll) == 4:
                    k = np.poly1d(np.polyfit(x, y, 10))
                    y = k(x)
                    plt.plot(x, y, label=strat)
                else:
                    plt.plot(x, y, dot, label=strat)
            plt.legend()
            if len(ll) == 2:  # do_full_sim:
                plt.savefig(plot_path / pathlib.Path('degree_comp_full.png'))
            else:
                plt.savefig(plot_path / pathlib.Path('degree_comp.png'))
            plt.show()

    # clustering
    if properties["clu"]:
        for ll in deg_clu_strats:
            if "full sim bitcoin" in ll:
                for strat in ll:
                    plt.title('Clustering of ' + strat)
                    plt.ylabel('clustering coefficient')
                    plt.ylim(0, 0.1)
                    plt.xlabel("node id")
                    y = DB[strat]["clu"]["dat"]
                    x = range(len(y))
                    if polate:
                        k = np.poly1d(np.polyfit(x, y, polate))
                        y = k(x)
                    if strat == "full sim ethereum":
                        plt.plot([], [])  # for color
                        plt.plot(x, y, dot, label=strat)
                    else:
                        plt.plot(x, y, dot, label=strat)

                    # plt.legend()
                    plt.savefig(plot_path / pathlib.Path('clustering_' + strat + '.png'))
                    plt.show()
            else:
                plt.title('Clustering')
                plt.ylabel('clustering coefficient')
                #plt.ylim(0, 0.18)
                plt.xlabel("node id")
                for strat in ll:
                    y = DB[strat]["clu"]["dat"]
                    x = range(len(y))
                    if True:
                        k = np.poly1d(np.polyfit(x, y, 10))
                        y = k(x)
                    plt.plot(x, y, label=strat)

                plt.legend()
                plt.savefig(plot_path / pathlib.Path('clustering_comp.png'))
                plt.show()

    # shortest paths
    if properties["pat"]:
        for ll in pat_clo_strats:
            if len(ll) == 4:  # do_all_but_full:
                plt.title('Shortest paths')
                plt.ylabel('node pairs')
                plt.xlabel("path length")
                y1 = list(DB["random bitcoin"]["pat"]["dat"])
                y2 = list(DB["lite sim bitcoin"]["pat"]["dat"])
                y3 = list(DB["random ethereum"]["pat"]["dat"])
                y4 = list(DB["lite sim ethereum"]["pat"]["dat"])
                l = max(len(y1), len(y2), len(y3), len(y4))
                while len(y1) < l:
                    y1.append(0)
                while len(y2) < l:
                    y2.append(0)
                while len(y3) < l:
                    y3.append(0)
                while len(y4) < l:
                    y4.append(0)
                x = np.arange(1, l + 1)
                plt.xticks(x)
                width = 0.2
                plt.bar(x - 0.3, y1, width=width, label="random bitcoin")
                plt.bar(x - 0.1, y2, width=width, label="lite sim bitcoin")
                plt.bar(x + 0.1, y3, width=width, label="random ethereum")
                plt.bar(x + 0.3, y4, width=width, label="lite sim ethereum")
                plt.legend()
                plt.savefig(plot_path / pathlib.Path('paths_comp.png'))
                plt.show()
            if len(ll) == 2:  # do_full_sim:
                plt.title('Shortest paths')
                plt.ylabel('node pairs')
                plt.xlabel("path length")
                y1 = list(DB["full sim bitcoin"]["pat"]["dat"])
                y2 = list(DB["full sim ethereum"]["pat"]["dat"])
                l = max(len(y1), len(y2))
                while len(y1) < l:
                    y1.append(0)
                while len(y2) < l:
                    y2.append(0)
                x = np.arange(1, l + 1)
                plt.xticks(x)
                width = 0.4
                plt.bar(x - 0.2, y1, width=width, label="full sim bitcoin")
                plt.bar(x + 0.2, y2, width=width, label="full sim ethereum")  # , color="m")
                plt.legend()
                plt.savefig(plot_path / pathlib.Path('paths_comp_full.png'))
                plt.show()

    # closeness
    if properties["clo"]:
        for ll in pat_clo_strats:
            plt.title('Closeness')
            plt.ylabel('closeness')
            plt.xlabel("node id")
            for strat in ll:
                y = DB[strat]["clo"]["dat"]
                x = range(len(y))
                if len(ll) == 4:
                    k = np.poly1d(np.polyfit(x, y, 10))
                    y = k(x)
                    plt.plot(x, y, label=strat)
                else:
                    plt.plot(x, y, dot, label=strat)
            plt.legend()
            if len(ll) == 2:
                plt.savefig(plot_path / pathlib.Path('closeness_comp_full.png'))
            else:
                plt.savefig(plot_path / pathlib.Path('closeness_comp.png'))
            plt.show()


def load_data():
    global DB
    with open(db_load_from, 'rb') as input:
        DB = _pickle.load(input)

########################################################################################################################
# functions to run the calculations and simulations
########################################################################################################################

def do_theoretical_calculations(strat, properties):
    st = theoretical_calculations.Calculations(N, strat, pathlength, steps)
    t = datetime.datetime.now()
    print("started " + strat + " at " + str(t.hour) + '_' + str(t.minute) + '_' + str(t.second))
    st.run(properties)
    t = datetime.datetime.now()
    print("finished at " + str(t.hour) + '_' + str(t.minute) + '_' + str(t.second))
    return st


def do_simulation(strat, properties):
    #random sim
    if strat in ["random bitcoin", "random ethereum"]:
        sr = randomNet.Simulation(n=N, strat=strat)
        t = datetime.datetime.now()
        print("started " + strat + " at " + str(t.hour) + '_' + str(t.minute) + '_' + str(t.second))
        sr.run(properties)
        t = datetime.datetime.now()
        print("finished at " + str(t.hour) + '_' + str(t.minute) + '_' + str(t.second))
        return sr

    # lite simulations
    if strat == "lite sim bitcoin":
        return simulate_bitcoin(lite=True)
    if strat == "lite sim ethereum":
        return simulate_ethereum(lite=True)

    # full simulations
    if strat == "full sim bitcoin":
        return simulate_bitcoin(lite=False)
    if strat == "full sim ethereum":
        return simulate_ethereum(lite=False)


def simulate_bitcoin(lite):
    stop_growing = N  # network stops growing if reached this amount of nodes
    n_iterations = round(2 * stop_growing)
    t_start = 1
    t_end = 8640000  # 100 day(s)
    # make sure that the outbound distribution is consistent with max_outbound_distribution
    outbound_distribution = 'const8_125'
    max_outbound_connections = 8
    connection_strategy = 'stand_bc'
    if lite:
        sd = TheorySimulationBitcoin.Simulation(simulation_type='bitcoin_protocol', with_evil_nodes=False,
                                           connection_strategy=connection_strategy,
                                           initial_connection_filter=False,
                                           outbound_distribution=outbound_distribution,
                                           data={'initial_min': -1, 'initial_max': -1},
                                           MAX_OUTBOUND_CONNECTIONS=max_outbound_connections,
                                           ti=ti)
    else:
        sd = bitcoinsimulationD.Simulation(simulation_type='bitcoin_protocol', with_evil_nodes=False,
                                           connection_strategy=connection_strategy,
                                           initial_connection_filter=False,
                                           outbound_distribution=outbound_distribution,
                                           data={'initial_min': -1, 'initial_max': -1},
                                           MAX_OUTBOUND_CONNECTIONS=max_outbound_connections,
                                           ti=ti)
    t = datetime.datetime.now()
    print("started bitcoin at " + str(t.hour) + '_' + str(t.minute) + '_' + str(t.second))
    sd.run(t_start=t_start, t_end=t_end, n_iterations=n_iterations, plot_first_x_graphs=0,
           avg_paths_after_n_iterations=[],
           MAX_OUTBOUND_CONNECTIONS=max_outbound_connections, numb_nodes=stop_growing)
    t = datetime.datetime.now()
    print("finished bitcoin at " + str(t.hour) + '_' + str(t.minute) + '_' + str(t.second))
    return sd


def simulate_ethereum(lite):
    stop_growing = N
    n_iterations = round(2 * stop_growing)
    t_start = 1
    t_end = 8640000  # 100 days
    outbound_distribution = 'const13_25'
    max_outbound_connections = 13
    if lite:
        s = TheorySimulationEthereum.Simulation(simulation_type='ethereum_protocol',
                                     with_evil_nodes=False,
                                     connection_strategy='stand_eth',
                                     initial_connection_filter=False,
                                     outbound_distribution=outbound_distribution,
                                     data={'initial_min': -1, 'initial_max': -1},
                                     MAX_OUTBOUND_CONNECTIONS=max_outbound_connections,
                                     ti=ti,
                                     )
    else:
        s = simulationEth.Simulation(simulation_type='ethereum_protocol',
                                     with_evil_nodes=False,
                                     connection_strategy='stand_eth',
                                     initial_connection_filter=False,
                                     outbound_distribution=outbound_distribution,
                                     data={'initial_min': -1, 'initial_max': -1},
                                     MAX_OUTBOUND_CONNECTIONS=max_outbound_connections,
                                     ti=ti)
    t = datetime.datetime.now()
    print("started ethereum at " + str(t.hour) + '_' + str(t.minute) + '_' + str(t.second))
    s.run(t_start=t_start, t_end=t_end, n_iterations=n_iterations, plot_first_x_graphs=0,
                   avg_paths_after_n_iterations=[], MAX_OUTBOUND_CONNECTIONS=max_outbound_connections,
                   numb_nodes=stop_growing)
    t = datetime.datetime.now()
    print("finished ethereum at " + str(t.hour) + '_' + str(t.minute) + '_' + str(t.second))
    return s

########################################################################################################################
if __name__ == "__main__":
    main()