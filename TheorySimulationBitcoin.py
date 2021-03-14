import datetime

import TheoryBitcoinNode as bn
from math import ceil
import networkanalyticsD as na
import networkx as nx
import numpy as np
import pprint
import random
import sys


class Simulation:
    simulation_protocol: str
    simulation_time: float
    DG_last_id: int

    def __init__(self, simulation_type='bitcoin_protocol', MAX_OUTBOUND_CONNECTIONS=8, with_evil_nodes=False,
                 show_plots=True, connection_strategy: str = 'p2c_max', initial_connection_filter: bool = False,
                 outbound_distribution='const8_125', data={}, ti=0):  # bitcoin_protocol power_2_choices
        self.MAX_OUTBOUND_CONNECTIONS = MAX_OUTBOUND_CONNECTIONS
        self.outbound_distribution = outbound_distribution
        self.evil_nodes_id = list()
        if with_evil_nodes:
            self.evil_nodes_percentage = .0025
        else:
            self.evil_nodes_percentage = 0
        self.data = data
        self.DG = nx.Graph()
        self.connection_strategy = connection_strategy
        self.DG_last_id = 0
        self.NUMBER_FIXED_DNS = self.MAX_OUTBOUND_CONNECTIONS
        self.FIXED_DNS = range(self.NUMBER_FIXED_DNS)
        self.simulation_time = 0.0  # seconds
        self.simulation_protocol = simulation_type
        self.whiteboard = na.NetworkAnalytics(self.DG, self.FIXED_DNS, show_plots=show_plots,
                                              connection_strategy=connection_strategy, with_evil_nodes=with_evil_nodes,
                                              max_outbound=MAX_OUTBOUND_CONNECTIONS,
                                              initial_connection_filter=initial_connection_filter,
                                              simulation_protocol=simulation_type,
                                              outbound_distribution=outbound_distribution,
                                              ti = ti)

        self._initialize_fixed_dns_servers()

        # if a node goes offline and online again it does not reconnect to the initial dns servers

        self.offline_nodes_reconnect = True
        self.offline_nodes = list()

    ############################
    # public functions
    ############################

    def run(self, t_start=1, t_end=60, n_iterations=124, plot_first_x_graphs=10,
            avg_paths_after_n_iterations=[10, 25, 50, 75, 100, 125, 150, 175, 200],
            MAX_OUTBOUND_CONNECTIONS=8, numb_nodes=3000, p_a=1):
        self.MAX_OUTBOUND_CONNECTIONS = MAX_OUTBOUND_CONNECTIONS
        finish_simulation_counter = 0
        p = p_a
        count_iterations = 0

        for ii, t in enumerate(list(np.linspace(t_start, t_end, n_iterations))):
            self.simulation_time = t
            count_iterations += 1
            print('simulation time: ' + str(round(self.simulation_time, 2))
                  + ', with ' + str(len(self.DG.nodes)) + ' nodes')
            #for node_id in self.DG.nodes:
                #self._process_envelopes(node_id)
            if finish_simulation_counter == 0:
                rand = random.randint(1, 100)
                if rand <= p*100:
                    #add a node
                    self._new_node_connects_to_network()
                    self._process_envelopes(self.DG_last_id)
                else:
                    #this node is node added, and we remove u.a.r a node
                    node_to_delete = 0
                    while(node_to_delete in self.FIXED_DNS):
                        node_set = list(self.DG.nodes)
                        node_to_delete = random.choice(node_set)
                        if len(self.DG.nodes) <= self.NUMBER_FIXED_DNS:
                            break
                    if node_to_delete not in self.FIXED_DNS:
                        neighbors = list(self.DG.nodes[node_to_delete][self.simulation_protocol].inbound.keys())
                        self._delete_node(node_to_delete, save_offline_node=self.offline_nodes_reconnect)
                        for i in neighbors:
                            self._process_envelopes(i)
                        self.DG_last_id += 1

            if (len(self.DG.nodes) >= numb_nodes) and (finish_simulation_counter == 0):
                #self.whiteboard.degree_in_time_Bitcoin(p=p, out=self.MAX_OUTBOUND_CONNECTIONS)
                return

    ############################
    # private functions
    ############################

    def _hacky_1(self) -> bool:
        for node in random.sample(self.DG.nodes(), round(len(self.DG.nodes()) * 0.1)):
            self.DG.nodes[node][
                self.simulation_protocol].MAX_OUTBOUND_CONNECTIONS = 600  # round(len(self.DG.nodes()) * 0.1)
            self.DG.nodes[node][self.simulation_protocol].MAX_TOTAL_CONNECTIONS = sys.maxsize

        ii = 0
        while True:
            ii += 1
            self.simulation_time += 0.6
            print('postprocess simulation time: ' + str(round(self.simulation_time, 2))
                  + ', iteration: ' + str(ii)
                  + ', with ' + str(len(self.DG.nodes)) + ' nodes')
            outbound_is_full = True
            for node_id in self.DG.nodes():
                self._process_envelopes(node_id)
                outbound_is_full = outbound_is_full and self.DG.nodes[node_id][
                    self.simulation_protocol].outbound_is_full()
            if outbound_is_full:
                return True

    def _process_envelopes(self, node_id):
        i = 0
        while self.MAX_OUTBOUND_CONNECTIONS != len(self.DG.nodes[node_id][self.simulation_protocol].outbound):
            if i==10:
                break
            envelopes = self.DG.nodes[node_id][self.simulation_protocol].interval_processes(
                self.simulation_time)
            tmp_envelope = list()
            for envelope in envelopes:
                # validate an envelope
                if envelope is None:
                    raise ValueError("envelope is None")
                if envelope['sender'] != node_id:
                    raise ValueError("envelope['sender'] = " + str(envelope['sender']))
                if envelope['receiver'] not in self.DG.nodes:
                    continue  # new connection question

                # parse envelope to check action
                if self.simulation_protocol == 'bitcoin_protocol':
                    if envelope['connect_as_outbound'] == 'can_I_send_you_stuff':
                        self._node_updates_outbound_connection(envelope['sender'], envelope=envelope)
                elif self.simulation_protocol == 'power_2_choices':
                    if envelope['whats_your_degree'] is True:
                        if len(tmp_envelope) == 1:
                            if tmp_envelope[0]['sender'] == envelope['sender']:
                                tmp_envelope.append(envelope)
                                self._node_updates_outbound_connection(envelope['sender'], envelope=tmp_envelope)
                            tmp_envelope = list()
                        else:
                            tmp_envelope.append(envelope)

                if envelope['get_address'] is True:
                    self._get_addresses_from_neighbour(envelope['sender'], envelope=envelope)
                elif envelope['kill_connection'] is True:
                    answer_envelope = self.DG.nodes[envelope['receiver']][self.simulation_protocol].receive_message(
                        self.simulation_time, envelope)
                    self.DG.nodes[answer_envelope['receiver']][self.simulation_protocol].receive_message(
                        self.simulation_time, answer_envelope)
                    if answer_envelope['connection_killed'] is True:
                        if self.DG.has_edge(envelope['sender'], envelope['receiver']):
                            self.DG.remove_edge(envelope['sender'], envelope['receiver'])
                        if self.DG.has_edge(envelope['receiver'], envelope['sender']):
                            self.DG.remove_edge(envelope['receiver'], envelope['sender'])
            i += 1

    def _offline_node_gets_online(self, node):
        id: int = node.get_id()
        self.DG.add_node(id)
        self.DG.nodes[id][self.simulation_protocol] = node
        self._get_addresses_from_neighbour(id)
        self._node_updates_outbound_connection(id)

    def _new_node_connects_to_network(self, show_network=False):
        self.DG_last_id += 1
        self.DG.add_node(self.DG_last_id)
        if float(len(self.evil_nodes_id) + 1) < len(self.DG.nodes) * self.evil_nodes_percentage:
            is_evil = True
        else:
            is_evil = False
        if self.simulation_protocol == 'bitcoin_protocol':
            max_outbound_connections, max_total_connections = self._get_outbound_connection_size()
            self.DG.nodes[self.DG_last_id][self.simulation_protocol] = bn.TheoryBitcoinNode(self.DG_last_id,
                                                                                     self.simulation_time,
                                                                                     self.FIXED_DNS,
                                                                                     self.DG,
                                                                                     MAX_OUTBOUND_CONNECTIONS=max_outbound_connections,
                                                                                     is_evil=is_evil,
                                                                                     connection_strategy=self.connection_strategy,
                                                                                     MAX_TOTAL_CONNECTIONS=max_total_connections)
            if is_evil:
                self.evil_nodes_id.append(self.DG_last_id)
                print('The evil node ' + str(self.DG_last_id) + ' has been added\n'
                                                                'all current evil nodes: ' + str(self.evil_nodes_id))
        elif self.simulation_protocol == 'power_2_choices':
            self.DG.nodes[self.DG_last_id][self.simulation_protocol] = p2c.Power2Choices(self.DG_last_id,
                                                                                        self.simulation_time,
                                                                                        self.FIXED_DNS)
        # self._get_addresses_from_neighbour(self.DG_last_id)
        self._initialize_outgoing_connections(self.DG_last_id)
        self._node_updates_outbound_connection(self.DG_last_id)
        if show_network is True:
            self.whiteboard.plot_net()

    def _get_outbound_connection_size(self) -> [int, int]:
        # returns the initial number of connections and the max number of connections
        if self.outbound_distribution is 'const8_125':
            return 8, 125
        if self.outbound_distribution is 'const13_25':
            return 13, 25
        if self.outbound_distribution is 'const8_inf':
            return 8, sys.maxsize
        if self.outbound_distribution is 'const13_inf':
            return 13, sys.maxsize
        if self.outbound_distribution is 'uniform_1_max':
            return random.randint(1, len(self.DG.nodes)), sys.maxsize
        if self.outbound_distribution is 'normal_mu8_sig4':
            return max(1, np.random.normal(8, 4, 1).astype(int)[0]), sys.maxsize
        if self.outbound_distribution is 'normal_mu_sig_auto':
            return max(1, np.random.normal(len(self.DG.nodes) * .25, len(self.DG.nodes) * .1, 1).astype(int)[0]), \
                   sys.maxsize
        if self.outbound_distribution is 'normal_mu16_sig8':
            return max(1, np.random.normal(16, 8, 1).astype(int)[0]), sys.maxsize
        if self.outbound_distribution is '1percent':
            if random.randint(1, 100) <= 1:
                return sys.maxsize / 2, sys.maxsize
            return self.MAX_OUTBOUND_CONNECTIONS, 125
        if self.outbound_distribution is '1percent_10':
            if random.randint(1, 100) <= 1:
                return len(self.DG.nodes) * 0.1, sys.maxsize
            return self.MAX_OUTBOUND_CONNECTIONS, 125
        if self.outbound_distribution is 'const_iter':
            return self.data['initial_min'], self.data['initial_max']
        if self.outbound_distribution is 'hacky_1':
            return self.MAX_OUTBOUND_CONNECTIONS, 125

        print(self.outbound_distribution)
        assert True

    def _delete_node(self, node_id, show_protocol=False, save_offline_node=True):

        envelopes = self.DG.nodes[node_id][self.simulation_protocol].go_offline(self.simulation_time)

        if show_protocol is True:
            pprint.pprint(envelopes)
        for envelope in envelopes:
            envelope_1 = self.DG.nodes[envelope['receiver']][self.simulation_protocol].receive_message(
                self.simulation_time, envelope)
            self.DG.nodes[envelope_1['receiver']][self.simulation_protocol].receive_message(self.simulation_time,
                                                                                           envelope_1)

        if save_offline_node is True:
            self.offline_nodes.append(self.DG.nodes[node_id][self.simulation_protocol])

        self.DG.remove_node(node_id)
        if node_id in self.evil_nodes_id:
            self.evil_nodes_id.remove(node_id)
        return True

    def _initialize_outgoing_connections(self, node_id):
        if self.DG.nodes[node_id][self.simulation_protocol]._hard_coded_dns is True:
            return
        dns_to_ask = random.sample(self.FIXED_DNS, k=2)
        for i in dns_to_ask:
            envelope_1 = self.DG.nodes[node_id][self.simulation_protocol].ask_neighbour_to_get_addresses(
                self.simulation_time, i)
            if envelope_1['receiver'] not in self.DG.nodes:
                continue
            envelope_2 = self.DG.nodes[envelope_1['receiver']][self.simulation_protocol].receive_message(
                self.simulation_time, envelope_1)
            envelope_3 = self.DG.nodes[envelope_2['receiver']][self.simulation_protocol].receive_message(
                self.simulation_time, envelope_2)
        return True

    def _get_addresses_from_neighbour(self, node_id, envelope=None, show_protocol=False):
        envelope_1 = envelope
        if envelope is None:
            envelope_1 = self.DG.nodes[node_id][self.simulation_protocol].ask_neighbour_to_get_addresses(
                self.simulation_time)

        if envelope_1['receiver'] not in self.DG.nodes:
            return False

        envelope_2 = self.DG.nodes[envelope_1['receiver']][self.simulation_protocol].receive_message(
            self.simulation_time, envelope_1)
        envelope_3 = self.DG.nodes[envelope_2['receiver']][self.simulation_protocol].receive_message(
            self.simulation_time, envelope_2)

        if show_protocol is True:
            pprint.pprint(envelope_1)
            pprint.pprint(envelope_2)
            pprint.pprint(envelope_3)
        return True

    def _node_updates_outbound_connection(self, node_id, envelope=None, show_protocol=False,
                                          show_connection_failures=False):

        if envelope is not None:
            envelope_1 = envelope
        else:
            envelope_1 = self.DG.nodes[node_id][self.simulation_protocol].update_outbound_connections(
                self.simulation_time)
        # there is no need to update the connections
        if envelope_1 is None:
            return False

        if self.simulation_protocol == 'bitcoin_protocol':
            success: bool = self._node_bitcoin(envelope_1, node_id, show_protocol, show_connection_failures)
        elif self.simulation_protocol == 'power_2_choices':
            success: bool = self._node_power_2_choices(envelope_1, node_id, show_protocol, show_connection_failures)
        return success

    def _node_power_2_choices(self, envelopes_1, node_id, show_protocol, show_connection_failures):
        assert len(envelopes_1) == 2, 'a node has to ask 2 other nodes in order to get to chose between two degrees'
        # if the node to connect does not exist anymore
        if envelopes_1[0]['receiver'] not in self.DG.nodes:
            if show_connection_failures:
                print('node_id: ' + str(node_id) + 'could not connect to ' + str(envelopes_1['receiver']))
            return False
        if envelopes_1[1]['receiver'] not in self.DG.nodes:
            if show_connection_failures:
                print('node_id: ' + str(node_id) + 'could not connect to ' + str(envelopes_1['receiver']))
            return False
        # one of the nodes wants to connect to himeself
        if (envelopes_1[0]['receiver'] == envelopes_1[0]['sender']) or (
                envelopes_1[1]['receiver'] == envelopes_1[1]['sender']):
            if show_connection_failures:
                print('node might connect to himself')
            return False
        envelopes_2 = list()
        envelopes_3 = list()
        for ii in range(2):
            envelopes_2.append(self.DG.nodes[envelopes_1[ii]['receiver']][self.simulation_protocol].receive_message(
                self.simulation_time, envelopes_1[ii]))
            envelopes_3.append(self.DG.nodes[envelopes_2[ii]['receiver']][self.simulation_protocol].receive_message(
                self.simulation_time, envelopes_2[ii]))
        envelope_4 = self.DG.nodes[envelopes_3[1]['receiver']][self.simulation_protocol].receive_message(
            self.simulation_time, envelopes_3[1])
        envelope_5 = self.DG.nodes[envelope_4['receiver']][self.simulation_protocol].receive_message(
            self.simulation_time, envelope_4)

        # add connection to graph if successful connection was established
        # propabaly still an error because we always return done!!!!
        if envelope_5['connect_as_outbound'] == 'done':
            self.DG.add_edge(envelopes_1[0]['sender'], envelopes_1[0]['receiver'])
        elif show_protocol:
            print(str(node_id) + ' could not update its outbound connections')
            print('### node ' + str(node_id) + ' looks for an outbound connection')
            pprint.pprint(envelopes_1)
            pprint.pprint(envelopes_2)
            pprint.pprint(envelopes_3)
            pprint.pprint(envelope_4)
            pprint.pprint(envelope_5)
            return False
        return True

    def _node_bitcoin(self, envelope_1, node_id, show_protocol, show_connection_failures):
        # if the node to connect does not exist anymore
        if envelope_1['receiver'] not in self.DG.nodes:
            if show_connection_failures:
                print('node_id: ' + str(node_id) + 'could not connect to ' + str(envelope_1['receiver']))
            return False

        # try to connect to node
        envelope_2 = self.DG.nodes[envelope_1['receiver']][self.simulation_protocol].receive_message(
            self.simulation_time, envelope_1)
        # envelope_2 is the answer_envelope created by the envelope_1['receiver']
        envelope_3 = self.DG.nodes[envelope_2['receiver']][self.simulation_protocol].receive_message(
            self.simulation_time, envelope_2)

        # add connection to graph if successful connection was established
        if envelope_3['connect_as_outbound'] == 'done':
            self.DG.add_edge(envelope_1['sender'], envelope_1['receiver'])
        elif show_protocol:
            print(str(node_id) + ' could not update its outbound connections')

        if show_protocol:
            print('### node ' + str(node_id) + ' looks for an outbound connection')
            pprint.pprint(envelope_1)
            pprint.pprint(envelope_2)
            pprint.pprint(envelope_3)
        return True

    def _initialize_fixed_dns_servers(self):
        for ii in self.FIXED_DNS:
            self.DG_last_id = ii
            self.DG.add_node(ii)
            max_outbound_connections, max_total_connections = self._get_outbound_connection_size()
            if self.simulation_protocol == 'bitcoin_protocol':
                self.DG.nodes[ii][self.simulation_protocol] = \
                    bn.TheoryBitcoinNode(ii, self.simulation_time, self.FIXED_DNS, self.DG,
                                                                            connection_strategy=self.connection_strategy,
                                          MAX_OUTBOUND_CONNECTIONS = max_outbound_connections,
                                         MAX_TOTAL_CONNECTIONS = max_total_connections)
            elif self.simulation_protocol == 'power_2_choices':
                self.DG.nodes[ii][self.simulation_protocol] = p2c.Power2Choices(ii, self.simulation_time, self.FIXED_DNS)
        for ii in list(self.DG.nodes):
            self._node_updates_outbound_connection(ii)

def pop_random_element_from_list(x):
    return x.pop(random.randrange(len(x)))


if __name__ == '__main__':
    t_start = 1
    t_end = 8640000  # 100 day(s)
    stop_growing = 100 # network stops growing if reached this amount of nodes
    n_iterations = 2 * stop_growing
    plot_first_x_graphs = 0
    avg_paths_after_n_iterations = []  # [0, 25, 50, 75, 100, 125, 150, 175, 200, 225]
    # outbound_distributions = ['hacky_1', 'const8_125', 'uniform_1_max', 'normal_mu8_sig4', '1percent',
    #                           'normal_mu_sig_auto', 'const13_125']
    # make sure that the outbound distribution is consistent with max_outbound_distribution
    outbound_distribution = 'const8_125'
    max_outbound_connections = 8
    # connection_strategy = ['stand_bc', 'p2c_min', 'p2c_max', 'geo_bc', 'no_geo_bc]
    connection_strategy = 'stand_bc'

    sd = Simulation(simulation_type='bitcoin_protocol', with_evil_nodes=False,
                    connection_strategy=connection_strategy,
                    initial_connection_filter=False,
                    outbound_distribution=outbound_distribution,
                    data={'initial_min': -1, 'initial_max': -1},
                    MAX_OUTBOUND_CONNECTIONS=max_outbound_connections,
                    ti=datetime.datetime.now())
    print("running bitcoin")
    sd.run(t_start=t_start, t_end=t_end, n_iterations=n_iterations, plot_first_x_graphs=plot_first_x_graphs,
           avg_paths_after_n_iterations=avg_paths_after_n_iterations,
           MAX_OUTBOUND_CONNECTIONS=max_outbound_connections, numb_nodes=stop_growing)

    w = sd.whiteboard
    #w.plot_degree()
    w.plot_degree_in_time(plotsolo=1, polate=0)
    print(sd.DG.degree)
    #w.betweenness()
    #w.betweenness_in_time()
    #w.clustering()
    w.clustering_in_time(plotsolo=1, polate=10)
    #w.shortest_path_histogram_undirected(plotsolo=1, is_final=True)
    w#.closeness_in_time(plotsolo=1)
    # w.known_addresses()
    #w.known_addresses_in_time()
    # w.minimum_edge_cut()
    #w.inout_sizes_in_time()

    print("finished bitcoin----------------")
