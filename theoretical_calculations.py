import matplotlib.pyplot as plt
import numpy as np
import math
import random
from mpl_toolkits.mplot3d import Axes3D

class Calculations:

    def __init__(self, N, strategy, pathlength, steps):
        self.N = N  # number of nodes
        if strategy == "theoretical bitcoin":
            self.strategy = "theoretical bitcoin"
            self.C = 125  # maximal total connections
            self.m = 8  # maximal outgoing connections
        else:  # ethereum
            self.strategy = "theoretical ethereum"
            self.C = 25  # maximal total connections
            self.m = 13  # maximal outgoing connections

        self.D = np.zeros((N + 1, N + 1))  # degree of node i at time t
        self.F = np.zeros(N + 1)  # number of full nodes at time t
        self.Cl = np.zeros(N + 1)  # clustering coefficient of node i at the end
        self.SP = np.zeros((N + 1, N + 1, 6))
        self.pathlength = pathlength  # max path length calculated
        self.avesp = np.zeros(self.pathlength + 1)  # average shortest path lengths oderso
        self.FP = np.zeros((N + 1, N + 1, self.pathlength + 1))  # tail distribution
        self.P = np.zeros((N + 1, N + 1, self.pathlength + 1))  # probability distribution
        self.expected_pathlength_all_pairs = np.zeros((self.N + 1, self.N + 1))
        self.closeness = np.zeros((self.N + 1))
        self.Pairwise_conn_prob = np.ones((self.N + 1, self.N + 1))
        self.ET = np.zeros(N + 1)  # E[T(i)]
        self.steps = steps
        self.P_alt = np.zeros((N + 1, N + 1, self.pathlength + 1))
        self.out = np.zeros((N+1))
        self.average_paths = list()

        if steps > 1:  # they are not sorted
            all_nodes = list(range(1, self.N+1))
            self.ii = sorted(random.choices(all_nodes, k=int(self.N/steps)))
            restricted_set = list(set(all_nodes) - set(self.ii))
            self.jj = sorted(random.choices(restricted_set, k=self.N // steps))
        else:
            self.ii = [*range(1, self.N+1)]
            self.jj = [*range(1, self.N+1)]


    def run(self, properties):
        if properties["deg"] or properties["pat"] or __name__ == "__main__":
            self.degree()

        if properties["clu"] or __name__ == "__main__":
            self.clustering()

        # self.shortest_paths()      # old, no good results
        # self.ave_shortest_paths()

        if __name__ == "__main__":
            self.calculate_pairs()
            self.paths()
            self.prob_distro()
            self.average_prob_distro()

        if __name__ == "__main__":
            self.calculate_expected_pathlength_all_pairs()
            self.calculate_closeness()

            #self.check_paths() #not necessary anymore :)

        # saving
        # np.savetxt("clustering_N="+str(self.N)+"_m="+str(self.m)+"_C="+str(self.C)+".csv", self.Cl, delimiter=", ", fmt='%3f')
        # np.savetxt("degrees_N=" + str(self.N) + "_m=" + str(self.m) + "_C=" + str(self.C) + ".csv", self.D[1:(self.N+1), -1],delimiter=", ", fmt='%3f')

    def get_degree_results(self, plotsolo=True, polate=0):
        degs = self.D[2:(self.N + 1), -1]
        avg = np.round(np.average(degs), 3)
        var = np.round(np.var(degs), 3)
        std = np.round(np.std(degs), 3)
        return degs, avg, var, std

    def get_clustering_results(self, plotsolo=True, polate=0):
        clu = self.Cl[2:self.N + 1]
        avg = np.round(np.average(clu), 3)
        var = np.round(np.var(clu), 3)
        std = np.round(np.std(clu), 3)
        return clu, avg, var, std

    def get_paths_results(self, plotsolo=True):
        """
        for i in self.ii:  # range(1, self.N, self.steps):
            for j in self.jj:  # range(i+1, self.N + 1, self.steps):
                pl = self.expected_pathlength_all_pairs[i][j]
                if pl in list(self.shortest_paths.keys()):
                    self.shortest_paths[pl] += 1
                else:
                    self.shortest_paths[pl] = 1
                    print("added length ", pl)
        """
        #avg = np.round(np.average(clu), 3)
        #var = np.round(np.var(clu), 3)
        #std = np.round(np.std(clu), 3)
        return self.average_paths


    def degree(self):
        for t in range(1, self.N + 1):
            self.F[t] = self.f(t)
            #self.out[t] = self.m * (t-1-self.F[t])/(t-1)
            for i in range(1, t + 1):
                self.D[i][t] = self.d_eth_new(i, t)
        # for i in range(1, self.m+1):
        #   self.D[i][self.N] += i-1

        # full nodes
        if __name__ == '__main__':
            plt.plot(range(self.N), self.F[1:])
            plt.title('# full nodes')
            plt.show()

        # degree
        if __name__ == '__main__':
            degs = self.D[2:(self.N + 1), -1]
            # x = range(self.N)
            # k = np.poly1d(np.polyfit(x, degs, 20))
            # ffit = k(x)
            # plt.plot(range(self.N), ffit)
            plt.plot(range(self.N - 1), degs)
            # plt.plot(range(2000), self.D[1:2001, 2000])
            # plt.plot(range(1000), self.D[1:1001, 1000])
            plt.title(str(self.strategy) + ' degree distribution')
            plt.xlabel('node id')
            plt.ylabel('degree')
            avg = np.round(np.average(degs), 3)
            std = np.round(np.std(degs), 3)
            var = np.round(np.var(degs), 3)
            var_std_text = "avg = " + str(avg) + "\n" + \
                           "std = " + str(std) + "\n" + \
                           "var = " + str(var) + "\n"
            plt.text(0.75, 0.9, var_std_text, horizontalalignment='left', verticalalignment='top',
                     transform=plt.gca().transAxes)
            plt.show()

    def clustering(self):
        for i in range(1, self.N + 1):
            print("clustering is at i=", i)
            self.Cl[i] = min(1, self.c(i))

        # clustering
        if __name__ == '__main__':
            clu = self.Cl[2:self.N + 1]
            # print(clu)
            plt.plot(range(self.N - 1), clu)
            plt.title('Clustering')
            plt.xlabel('node id')
            plt.ylabel('clustering coefficient')
            avg = np.round(np.average(clu), 6)
            std = np.round(np.std(clu), 6)
            var = np.round(np.var(clu), 6)
            var_std_text = self.strategy + "\n" + \
                           "# nodes = " + str(self.N) + "\n" + \
                           "avg = " + str(avg) + "\n" + \
                           "std = " + str(std) + "\n" + \
                           "var = " + str(var) + "\n"

            plt.text(0.75, 0.9, var_std_text, horizontalalignment='left', verticalalignment='top',
                     transform=plt.gca().transAxes)
            plt.show()

    def f(self, t):
        sol = 0
        for i in range(1, t):
            if self.D[i][t - 1] >= self.C:
                sol += 1
        return sol

    def d_eth_new(self, i, t):
        if i > 1:
            return min(self.m * (np.log((t - 1) / (i - 1)) + (i - 1 - self.F[i]) / (i - 1)), self.C)
        else:
            return self.C

    def d_retries(self, i, t):
        if i > 1:
            return min(self.m * (np.log((t - 1) / (i - 1)) + (i - 1 - self.F[i]) / (i - 1)) + self.aux(i, t), self.C)
        else:
            return self.C

    def aux(self, i, t):
        sol = 0
        for j in range(1, t):
            if j == i:
                continue
            if j > self.F[t]:
                sol += (self.m - self.out[j]) * np.log(t-1)
                if self.out[j] < self.m:
                    self.out[j] += (self.m - self.out[j]) * (t-1-self.F[t])/(t-1)
        for j in range(1, i):
            if j == i:
                continue
            if j > self.F[i]:
                sol -= (self.m - self.out[i]) * np.log(i-1)
                self.out[j] -= (self.m - self.out[j]) * (t - 1 - self.F[t]) / (t - 1)
        return sol

    def c(self, i):
        return (2 * self.triangles(i)) / (self.D[i][self.N] * (self.D[i][self.N] - 1))

    def triangles(self, i):
        sol1 = 0
        sol2 = 0
        sol3 = 0
        # i<j<k
        for j in range(i + 1, self.N):
            sol11 = 0
            for k in range(j + 1, self.N + 1):
                if i > self.F[k] and j > self.F[k]:
                    sol11 += 1 / ((k - 1) * (k - 2))
            if i > self.F[j]:
                sol1 += 1 / (j - 1) * sol11

        # j<i<k
        if i > 1:
            for j in range(1, i):
                if j <= self.F[i]:
                    continue
                for k in range(i + 1, self.N + 1):
                    if j > self.F[k] and i > self.F[k]:
                        sol2 += 1 / ((i-1) * (k - 1) * (k - 2))

        # j<k<i
        if i > 2:
            for j in range(1, i - 1):
                if j <= self.F[i]:
                    continue
                for k in range(j + 1, i):
                    if j > self.F[k] and k > self.F[i]:
                        sol3 += 1 / ((k - 1)*(i - 1) * (i - 2))

        return self.m ** 2 * (self.m - 1) * (sol1 + sol2 + sol3)

    def calculate_pairs(self):
        for i in range(1, self.N):  # should always be over full range
            for j in range(i+1, self.N+1):
                if not (i <= self.m and j <= self.m):  # for j <= m it is already 1 by instantiation
                    if self.strategy == "theoretical ETH":
                        if i < self.F[j]:  # i is full when j tries to connect to it
                            self.Pairwise_conn_prob[i][j] = 0  # self.m / (j - 1) * (j-1-self.F[j])/(j-1)
                            self.Pairwise_conn_prob[j][i] = self.Pairwise_conn_prob[i][j]
                        else:
                            self.Pairwise_conn_prob[i][j] = self.m / (j - 1)
                            self.Pairwise_conn_prob[j][i] = self.m / (j - 1)
                    else:
                        self.Pairwise_conn_prob[i][j] = self.m / (j-1)
                        self.Pairwise_conn_prob[j][i] = self.m / (j-1)

    def paths(self):  # calculates the shortest path length tail distribution
        for i in self.ii:  # range(1, self.N, self.steps):
            for j in self.jj:  # range(i+1, self.N + 1, self.steps):
                self.FP[i][j][0] = 1
                self.FP[i][j][1] = self.q(i, j)
                self.FP[j][i][0] = 1
                self.FP[j][i][1] = self.FP[i][j][1]
                print("paths at (i, j) = (", i, ", ", j, ")")
                for k in range(2, self.pathlength+1):
                    self.FP[i][j][k] = self.FP[i][j][k-1] * self.condprob(i, j, k, [i, j])
                    self.FP[j][i][k] = self.FP[i][j][k]

    def condprob(self, i, j, k, outtakes):  # calculates the recusrive conditional probabilities
        if k == 1:
            return self.q(i, j)
        else:
            sol = 1
            for l in range(1, self.N + 1):
                if l in outtakes:
                    continue
                else:
                    outtakes.append(l)
                    sol *= (self.q(i, l) + (self.p(i, l) * self.condprob(l, j, k - 1, outtakes)))
            return sol

    def q(self, i, j):  # opposite of p
        return 1 - self.Pairwise_conn_prob[i][j]

    def p(self, i, j):  # probability that i and j are connected
        return self.Pairwise_conn_prob[i][j]

    def prob_distro(self):  # computes probability density
        for i in self.ii:  # range(1, self.N, self.steps):
            for j in self.jj:  # range(i + 1, self.N + 1, self.steps):
                helper = 0
                for k in range(1, self.pathlength+1):
                    self.P[i][j][k] = self.FP[i][j][k - 1] - self.FP[i][j][k]
                    self.P[j][i][k] = self.P[i][j][k]
                    #helper += self.P[i][j][k]
                #self.P[i][j][self.pathlength] = 1 - helper
                #self.P[j][i][self.pathlength] = self.P[i][j][k]

    def average_prob_distro(self):
        avg = 0

        for k in range(1, self.pathlength + 1):
            self.average_paths.append(0)
            for i in self.ii:  # range(1, self.N, self.steps):
                for j in self.jj:  # range(i+1, self.N + 1, self.steps):
                    if i == j:
                        continue
                    self.average_paths[k-1] += self.P[i][j][k]
            if self.steps == 1:
                div = self.N*(self.N-1)
            else:
                div = len(self.ii)*(len(self.jj))
            self.average_paths[k-1] /= div
            avg += k*self.average_paths[k-1]  # this average does not account for paths of length > self.pathlength
        print("average prob distro = ", self.average_paths)
        if __name__ == '__main__':
            plt.bar(range(1, self.pathlength+1), self.average_paths)
            plt.xticks(range(1, self.pathlength+1))
            plt.title("average probability distribution of pathlength")
            avg = np.round(avg, 3)
            var_std_text = self.strategy + "\n" + \
                           "# nodes = " + str(self.N) + "\n" + \
                           "% pairs = " + str(100/(self.steps*self.steps)) + "\n" + \
                           "avg = " + str(avg) + "\n"
            plt.text(0.75, 0.9, var_std_text, horizontalalignment='left', verticalalignment='top',
                     transform=plt.gca().transAxes)
            plt.xlabel('shortest path length')
            plt.ylabel('% node pairs')
            plt.show()
        return avg, self.average_paths

    def calculate_expected_pathlength_all_pairs(self):  # is imprecise when computed max pathlength is too low
        avg = 0
        for i in self.ii:  # range(1, self.N, self.steps):
            for j in self.jj:  # range(i+1, self.N + 1, self.steps):
                for k in range(1, self.pathlength + 1):
                    self.expected_pathlength_all_pairs[i][j] += k * self.P[i][j][k]
                    avg += k * self.P[i][j][k]
        avg /= len(self.ii)*(len(self.jj))

        average_probs = list()
        for i in self.ii:  # range(1, self.N, self.steps):
            for j in self.jj:  # range(i+1, self.N + 1, self.steps):
                average_probs.append(self.expected_pathlength_all_pairs[i][j])

        if __name__ == "__main__":
            print("average expected path lengths = ", average_probs)
            plt.plot(range(len(average_probs)), average_probs)
            #plt.xticks(range(1, self.pathlength + 1))
            plt.title("average expected path lengths")
            avg = np.round(avg, 3)
            var_std_text = self.strategy + "\n" + \
                           "# nodes = " + str(self.N) + "\n" + \
                           "% pairs = " + str(self.steps) + "\n" + \
                           "avg = " + str(avg) + "\n"
            plt.text(0.75, 0.9, var_std_text, horizontalalignment='left', verticalalignment='top',
                     transform=plt.gca().transAxes)
            plt.ylabel('expected shortest path length')
            plt.ylabel('node pair')
            plt.show()

        #Axes3D.plot_surface(self.ii, self.jj, average_probs)
        #plt.show()

        print("avg expected pathlength = ", avg)

    def calculate_closeness(self):
        for i in range(1, self.N + 1, self.steps):
            for j in range(1, self.N + 1, self.steps):
                if i != j:
                    if i < j:
                        self.closeness[i] += self.expected_pathlength_all_pairs[i][j]
                    else:
                        self.closeness[i] += self.expected_pathlength_all_pairs[j][i]
            self.closeness[i] = (self.N - 1)/self.steps / self.closeness[i]
        x = range(1, self.N + 1)
        #k = np.poly1d(np.polyfit(x, self.closeness[1:], 2))
        #y = k(x)
        plt.plot(x, self.closeness[1:])
        avg = np.round(np.average(self.closeness[1:]), 3)
        std = np.round(np.std(self.closeness[1:]), 3)
        var_std_text = self.strategy + "\n" + \
                       "# nodes = " + str(self.N) + "\n" + \
                       "avg = " + str(avg) + "\n" + \
                       "std = " + str(std) + "\n"
        plt.text(0.75, 0.9, var_std_text, horizontalalignment='left', verticalalignment='top',
                 transform=plt.gca().transAxes)
        # plt.xlim(0.5, len(summary_of_shortest_paths) - 0.5)
        # plt.xticks(hops, hops)
        plt.xlabel('node id')
        plt.ylabel('closeness')
        plt.title('Closeness')
        if __name__ == "__main__":
            plt.show()

    def check_paths(self):
        # other computation of pathlengths 1 and 2
        for i in range(1, self.N, self.steps):
            for j in range(i+1, self.N + 1, self.steps):
                self.P_alt[i][j][1] = self.p(i, j)
                print("alt paths at (i, j) = (", i, ", ", j, ")")
                sol = 1  # Xijk=1
                #sol = 0  # Xijk = 0
                for k in range(1, self.N+1):
                    if not k == i or k == j:
                        sol *= self.xijk0(i,j,k)  # here we calculate prob Xijk = 0
                        #sol *= self.xijk0(i, j, k)
                self.P_alt[i][j][2] = (1 - sol) * (1-self.P[i][j][1])  # Xijk = 0
                #self.P_alt[i][j][2] = sol * (1 - self.P[i][j][1])  # Xijk = 1
        avg_alt, average_probs_alt = self.average_prob_distro_alter()
        avg, average_probs = self.average_prob_distro()
        print("recursion: ", avg, ", ", average_probs, "\n", "alternative: ",  avg_alt, ", ", average_probs_alt)

    def xijk0(self, i, j, k):
        if i < j < k:
            if j <= self.m or k <= self.m:
                return 0
            else:
                return 1 - (self.p(i, k) * (self.m-1)/(k-2))
        elif i < k < j:
            if j <= self.m:
                return 0
            else:
                return 1 - (self.p(i, k) * self.m/(j-2))
        else:
            if j <= self.m:
                return 0
            else:
                if i <= self.m:
                    return 1 - self.m/(j-2)
                else:
                    return 1 - (self.m/(i-2) * self.m/(j-2))

    def xijk1(self, i, j, k):
        if i < j < k:
            if j <= self.m:
                return 0
            else:
                return (self.p(i, k) * (self.m-1)/(k-2))
        elif i < k < j:
            if j <= self.m:
                return 0
            else:
                return (self.p(i, k) * self.m/(j-2))
        else:
            if j <= self.m:
                return 0
            else:
                if i <= self.m:
                    return self.m/(j-2)
                else:
                    return (self.m/(i-2) * self.m/(j-2))

    def average_prob_distro_alter(self):
        avg = 0
        average_probs = list()
        for k in range(1, self.pathlength + 1):
            average_probs.append(0)
            for i in range(1, self.N, self.steps):
                for j in range(i + 1, self.N + 1, self.steps):
                    average_probs[k - 1] += self.P_alt[i][j][k]
            average_probs[k - 1] /= (self.N * (self.N - 1) / 2)
            avg += k * average_probs[k - 1]
        return avg, average_probs

##### old stuff #####
"""

    def d_btc(self, i,t):
        return min(self.m + self.m * np.log((t - self.F[t]) / (i - self.F[i])), self.C)

    def d_eth(self, i, t):
        if i > 1:
            if 13 > ((t-1)-self.F[t]) / (t-1) * 22:
                return min(22 * np.log((t-1)/(i-1)) + (i-1-self.F[i]) / (i-1), self.C)
            else:
                return min(13 * np.log((t - 1 - self.F[t]) / (i - 1 - self.F[i])) + 13, self.C)
        else:
            return 25

    def d_eth_old(self, i, t):
        if i > 1:
            if 13 > ((t-1)-self.F[t]) / (t-1) * 22:
                return min(22 * np.log((t-1)/(i-1)) + (i-1-self.F[i]) / (i-1), self.C)
            else:
                return min(22 * np.log((t - 1) / (i - 1)) + 13, self.C)
        else:
            return 25

    def shortest_paths_0(self):
        #length 1
        for i in range(self.m + 2, self.N):
            for j in range(i+1, self.N + 1):
                self.SP[i][j][1] = self.m/(j-1)  # change this to N(j) if with dying nodes
        # lengths >1
        for l in range(2, len(self.avesp)):
            for i in range(self.m + 2,self.N):
                for j in range(i+1,self.N+1):
                    sol = 0
                    for k in range(self.m + 2, self.N+1):
                        if k == i or k == j:
                            continue
                        else:
                            if i < k:
                                if k < j:
                                    sol += self.SP[i][k][l-1] * self.SP[k][j][1]
                                else:
                                    sol += self.SP[i][k][l-1] * self.SP[j][k][1]
                            else:
                                if k < j:
                                    sol += self.SP[k][i][l-1] * self.SP[k][j][1]
                                else:
                                    sol += self.SP[k][i][l-1] * self.SP[j][k][1]
                    sol *= (1-self.SP[i][j][l-1]) / (self.N - 2)
                    self.SP[i][j][l] = sol

    def ave_shortest_paths_0(self):
        for l in range(1, len(self.avesp)):
            for i in range(self.m + 2, self.N):
                for j in range(i+1, self.N+1):
                    self.avesp[l] += self.SP[i][j][l] * 2
            self.avesp[l] = self.avesp[l] / (self.N * (self.N-1))

        print(self.avesp[1:])
        plt.bar(range(1,len(self.avesp)), self.avesp[1:], width=0.8, bottom=None, align='center')
        plt.title("average shortest paths")
        plt.show()
"""

if __name__ == '__main__':
    strategies = ["etheoretical bitcoin"]  # for bitcoin it must be "theoretical bitcoin", for ethereum anything else
    properties = {"deg": 1, "clu": 0, "path": 1, "close": 0}  # choose which properties you want calculated
    for s in strategies:
        c = Calculations(N=300, strategy=s, pathlength=5, steps=10)
        c.run(properties)
