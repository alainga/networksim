import datetime
import networkanalyticsD as na
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np


class Simulation:

    def __init__(self, n=1000, strat="random bitcoin"):
        self.DG = nx.Graph()
        self.N = n
        self.nodes = dict()
        self.strat = strat
        if self.strat == "random bitcoin":
            self.C = 125
            self.m = 8
        else:  # ethereum
            self.C = 25
            self.m = 13
        od = str(self.m)+"_"+str(self.C)
        self.whiteboard = na.NetworkAnalytics(self.DG, show_plots=True, connection_strategy=strat, with_evil_nodes=False,
                                         max_outbound=self.m, simulation_protocol=self.strat, outbound_distribution=od,
                                         hard_coded_dns=range(0), ti=datetime.datetime.now())

    def run(self, properties):
        for i in range(1, self.m):
            self.DG.add_node(i)
            for j in range(i+1, self.m+1):
                self.DG.add_edge(i, j)
        for i in range(self.m+1, self.N+1):
            print("adding node ", i)
            self.DG.add_node(i)
            for k in range(self.m):
                candidate = random.choice(list(set(self.DG.nodes.keys()) - set([i])))
                if self.DG.has_edge(i, candidate):
                    k -= 1
                    continue
                if not self.is_full(candidate):
                    self.DG.add_edge(i, candidate)

    def is_full(self, id):
        if self.DG.degree(id) >= self.C:
            return True
        return False


if __name__ == '__main__':
    """
    sr = Simulation(n=3000, ti=datetime.datetime.now(), strat="b")
    print("running randomNet")
    sr.run()

    w = sr.whiteboard
    #w.plot_degree()
    w.plot_degree_in_time()
    w.degree_in_time_Bitcoin(1, 8)
    #w.betweenness()
    #w.clustering()
    w.shortest_path_histogram_undirected(is_final=True)
    #w.closeness()
    #w.closeness_in_time()

    print("finished randomNet----------------")
    """
    n = 300
    iterations = 1
    properties = {"deg": 1, "clu": 0, "path": 1, "close": 1}  # choose which properties you want calculated
    shortest_paths_all_pairs_all_iterations = dict()
    average_shortest_paths_all_pairs = np.zeros((n+1, n+1))
    summary_of_shortest_paths = np.zeros((7))
    closeness = np.zeros((n+1))
    average_shortest_path = 0
    strategy = "eth"
    for i in range(iterations):
        sr = Simulation(n=n, strat=strategy)  #took out ti=datetime.datetime.now()
        print("running randomNet ", i)
        sr.run(properties)
        w = sr.whiteboard
        w.plot_degree_in_time()
        w.shortest_path_histogram_undirected(is_final=True)
        w.closeness_in_time()
        if False:
            shortest_paths_all_pairs_all_iterations[i] = dict(nx.all_pairs_shortest_path_length(sr.DG))
            for start_node, values in shortest_paths_all_pairs_all_iterations[i].items():
                for end_node, path_length in values.items():
                    average_shortest_paths_all_pairs[start_node][end_node] += path_length / iterations
                    summary_of_shortest_paths[path_length] += 1/iterations
    if False:
        helper = list()
        for i in range(1, len(summary_of_shortest_paths)):
            #helper.append(summary_of_shortest_paths[i]*i)
            average_shortest_path += summary_of_shortest_paths[i]*i
        average_shortest_path /= n*(n-1)
        print("average shortest path = ", average_shortest_path)

    # closeness
    if False:
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i == j:
                    continue
                else:
                    closeness[i] += average_shortest_paths_all_pairs[i][j]
            closeness[i] = (n-1) / closeness[i]

    # plotting
    # hops
    if False:
        print("summary of shortest paths = ", summary_of_shortest_paths)
        plt.bar(range(len(summary_of_shortest_paths)), summary_of_shortest_paths)
        plt.xlim(0.5, len(summary_of_shortest_paths)-0.5)
        avg = np.round(average_shortest_path, 3)
        # std =
        var_std_text = "random_" + str(strategy) + "\n" + \
                       "# nodes = " + str(n) + "\n" + \
                       "avg = " + str(avg) + "\n"
        plt.text(0.75, 0.9, var_std_text, horizontalalignment='left', verticalalignment='top',
                 transform=plt.gca().transAxes)
        #plt.xticks(hops, hops)
        plt.xlabel('path length')
        plt.ylabel('number of node pairs')
        plt.title('Average shortest paths over ' + str(iterations) + " networks")
        plt.show()

    # closness
    if False:
        x = range(n)
        k = np.poly1d(np.polyfit(x, closeness[1:], 2))
        y = k(x)
        plt.plot(x, y)
        avg = np.round(np.average(closeness[1:]), 3)
        std = np.round(np.std(closeness[1:]), 3)
        var_std_text = strategy + "\n" + \
                       "# nodes = " + str(n) + "\n" + \
                       "avg = " + str(avg) + "\n" + \
                       "std = " + str(std) + "\n"
        plt.text(0.75, 0.9, var_std_text, horizontalalignment='left', verticalalignment='top',
                 transform=plt.gca().transAxes)
        #plt.xlim(0.5, len(summary_of_shortest_paths) - 0.5)
        # plt.xticks(hops, hops)
        plt.xlabel('node id')
        plt.ylabel('closeness')
        plt.title('Closeness')
        plt.show()

