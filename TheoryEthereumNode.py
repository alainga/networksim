from collections import Counter
import random
import math


class EthereumConstant(object):

    # how old addresses can maximally be
    ADDRMAN_HORIZON_DAYS = 30

    # how recent a successful connection should be before we allow an address to be evicted from tried
    ADDRMAN_REPLACEMENT_HOURS = 4

    # Ethereum only maintains 16 buckets
    # because it's very unlikely that a node will ever encounter a node that's closer
    P2P_BUCKET_SIZE = 16

class TheoryEthereumNode(EthereumConstant):
    id: int
    ############################
    # initialization
    ############################
    def __init__(self, node_id: int, DNS_rnd: dict , t: float, hard_coded_dns, DG: object, MAX_OUTBOUND_CONNECTIONS: int = 13,
                 MAX_TOTAL_CONNECTIONS: int = 25,
                 is_evil: bool = False, connection_strategy: str = 'stand_eth') -> object:

        #we choose a 32-bit number which corresponds to ethereum's 256-bit node_id
        if node_id in hard_coded_dns:
            self.rnd_id = DNS_rnd[node_id]
        else:
            self.rnd_id = format(random.randint(0, 2 ** 32), '032b')
        self.node_id_to_rnd = dict()
        self.id = node_id
        self.DNS_rnd = DNS_rnd
        self.DG = DG

        # bucket contains nodes, ordered by their last activity
        self.bucket = dict()
        for i in range(0, self.P2P_BUCKET_SIZE):
            self.bucket[i] = []
        self.outbound_is_full = False
        self.connection_strategy = connection_strategy
        self.is_evil = is_evil
        if is_evil:
            # self.MAX_OUTBOUND_CONNECTIONS = int(sys.maxsize / 2)
            #self.MAX_TOTAL_CONNECTIONS = sys.maxsize
            assert True
        else:
            self.MAX_OUTBOUND_CONNECTIONS = MAX_OUTBOUND_CONNECTIONS
            self.MAX_TOTAL_CONNECTIONS = MAX_TOTAL_CONNECTIONS

        # mapping from node to timestamp
        self.addrMan = dict()
        for address in hard_coded_dns:
            if address is not node_id:
                self.addrMan[address] = t
                self._dns_to_bucket(address)
                self.node_id_to_rnd[address] = self.DNS_rnd[address]

        # peer is an outbound connection
        self.outbound = dict()
        # peer is an inbound connection
        self.inbound = dict()
        # peer knows about address (can set/get)
        self.address_known = dict()
        # set of addresses to send to peer
        self.address_buffer = list()

        # bounce message from a call
        self.output_envelope = list()

        #record the node_id of the neighbors chosen either from RandomNodes or lookupBuf
        self.record_outgoing_conn = dict()
        self.record_outgoing_conn['RandomNodes'] = []
        self.record_outgoing_conn['lookupBuf'] = []

        # the most recent discovered peers, dict from node_id to rnd_id
        self.lookupBuf = dict()

        # when the addresses were broadcasted the last time
        self.interval_timestamps = {'30min': t}

        self._hard_coded_dns = hard_coded_dns
        if self.id in self._hard_coded_dns:
            self._is_hard_coded_DNS = True
        else:
            self._is_hard_coded_DNS = False

    ############################
    # public functions
    ############################

    def get_id(self) -> int:
        return self.id

    def is_hard_coded_dns(self):
        return self._is_hard_coded_DNS

    def number_outbound(self):
        return len(self.outbound)

    def outbound_is_full(self) -> bool:
        if len(list(self.outbound.keys())) + len(list(self.inbound.keys())) >= self.MAX_OUTBOUND_CONNECTIONS:
            if self.outbound_is_full_bool is False:
                self.outbound_is_full_bool = True
        return self.outbound_is_full_bool

    def update_outbound_connections(self, t):
        address = self._get_address_to_connect()

        # look from a global perspective on the code and find out which degree out of two possibilities one should get
        if (self.connection_strategy is 'p2c_max') or (self.connection_strategy is 'p2c_min'):
            address2 = self._get_address_to_connect()
            if (address is not None) and (address2 is not None):
                if self.DG.is_directed():
                    graph = self.DG.to_undirected()
                else:
                    graph = self.DG
                addresses_in_graph = True
                if address not in graph.node:
                    addresses_in_graph = False
                if address2 not in graph.node:
                    addresses_in_graph = False
                if addresses_in_graph:
                    degree1 = graph.degree(address)
                    degree2 = graph.degree(address2)
                    if self.connection_strategy is 'p2c_min':
                        address = address if degree1 <= degree2 else address2
                    elif self.connection_strategy is 'p2c_max':
                        address = address if degree1 >= degree2 else address2
        elif self.connection_strategy is 'geo_bc':
            numb_bubble = 5
            my_bubble = self.id % numb_bubble
            my_bubble_neighbours = [x % numb_bubble for x in self.outbound.keys()]
            for _ in range(40):
                if address is None:
                    break
                address_bubble = address % numb_bubble
                if address_bubble not in my_bubble_neighbours:
                    # a new bubble that we have not yet connected
                    break
                if address_bubble == my_bubble:
                    # connecting to the same bubble
                    if Counter(my_bubble_neighbours)[my_bubble] <= 8:
                        # there are less than x connections within my own bubble
                        break
                address = self._get_address_to_connect()


        if address is None:
            return address
        envelope = self._get_empty_envelope(t, address)
        envelope['connect_as_outbound'] = 'can_I_send_you_stuff'
        self.output_envelope = [envelope]
        return envelope

    def ask_for_outbound_connection(self, t, address):
        if len(self.inbound) >= self.MAX_TOTAL_CONNECTIONS - self.MAX_OUTBOUND_CONNECTIONS:
            return False
        self.inbound[address] = t
        return True

    def go_offline(self, t):
        connected_nodes = list(set(list(self.inbound) + list(self.outbound)))
        envelopes = [self._kill_connection(t, address) for address in connected_nodes]
        self.output_envelope = envelopes
        self.record_outgoing_conn['RandomNodes'].clear()
        self.record_outgoing_conn['lookupBuf'].clear()
        return envelopes

    def receive_message(self, t, envelope):
        sender_rnd_id = list(envelope['sender'].values()).pop()
        sender_node_id = list(envelope['sender'].keys()).pop()
        # validate input
        if sender_node_id == self.id:
            raise ValueError("envelope['sender'] = " + str(envelope['sender']) + ", self.id = " + str(self.id))
        if sender_node_id == envelope['receiver']:
            raise ValueError('envelope is sent to itself')

        # initialize return statement
        answer_envelope = self._get_empty_envelope(t, sender_node_id)

        # sender has never been seen before
        if sender_node_id not in self.addrMan:
            self.addrMan[sender_node_id] = t
        if sender_node_id not in self.address_known:
            self.address_known[sender_node_id] = dict()
        if sender_node_id not in self.node_id_to_rnd.keys():
            self.node_id_to_rnd[sender_node_id] = sender_rnd_id
            self._insert_into_bucket(sender_node_id)
            self._add_to_lookupBuf(sender_node_id, sender_rnd_id)
        # address message response from connect_as_outbound
        if envelope['connect_as_outbound'] == 'can_I_send_you_stuff':
            if self.ask_for_outbound_connection(t, sender_node_id):
                answer_envelope['connect_as_outbound'] = 'accepted'
            else:
                answer_envelope['connect_as_outbound'] = 'rejected'
        if envelope['connect_as_outbound'] == 'accepted':
            self.outbound[sender_node_id] = t
            #if sender_node_id not in self.record_outgoing_conn['RandomNodes']:
               # self.record_outgoing_conn['RandomNodes'].insert(0, sender_node_id)
            answer_envelope['connect_as_outbound'] = 'done'
        elif envelope['connect_as_outbound'] == 'rejected':
            if sender_node_id in self.lookupBuf:
                del self.lookupBuf[sender_node_id]
            if sender_node_id in self.record_outgoing_conn['RandomNodes']:
                self.record_outgoing_conn['RandomNodes'].remove(sender_node_id)
            if sender_node_id in self.record_outgoing_conn['lookupBuf']:
                self.record_outgoing_conn['lookupBuf'].remove(sender_node_id)

        # neighbour node goes offline
        if envelope['kill_connection'] is True:
            if sender_node_id in self.record_outgoing_conn['RandomNodes']:
                self.record_outgoing_conn['RandomNodes'].remove(sender_node_id)
            if sender_node_id in self.record_outgoing_conn['lookupBuf']:
                self.record_outgoing_conn['lookupBuf'].remove(sender_node_id)
            if sender_node_id in self.outbound:
                self.outbound.pop(sender_node_id, None)
            if sender_node_id in self.inbound:
                self.inbound.pop(sender_node_id, None)
            answer_envelope['connection_killed'] = True
        if envelope['connection_killed'] is True:
            if sender_node_id in self.outbound:
                self.outbound.pop(sender_node_id, None)
            if sender_node_id in self.inbound:
                self.inbound.pop(sender_node_id, None)
            if sender_node_id in self.record_outgoing_conn['RandomNodes']:
                self.record_outgoing_conn['RandomNodes'].remove(sender_node_id)
            if sender_node_id in self.record_outgoing_conn['lookupBuf']:
                self.record_outgoing_conn['lookupBuf'].remove(sender_node_id)
            #self.record_outgoing_conn['RandomNodes'].clear()
            #self.record_outgoing_conn['lookupBuf'].clear()

        # update address timestamps
        if sender_node_id in self.addrMan:
            if sender_node_id in self.outbound:
                if self.addrMan[sender_node_id] < t - 20 * 60:
                    self.addrMan[sender_node_id] = t

        # address message from peer with addresses in address_vector
        envelopes = []
        asked = list()
        asked.insert(0, sender_node_id)
        iter = envelope['address_list'].copy()
        address_dict = iter
        address_list = dict()
        if len(iter) > 0:
            address_list = iter.pop()
        for address, rnd_id in address_list.items():
            if address != self.id:
                if address not in self.node_id_to_rnd.keys():
                    # we have received new nodes, so ask them for even closer addresses
                    envelope_1 = self._get_empty_envelope(t, address, envelope['random_address'])
                    envelope_1['get_address'] = True
                    envelopes.append(envelope_1)
                    answer_envelope = envelopes
                    self._add_to_lookupBuf(address, rnd_id)
                    self.node_id_to_rnd[address] = rnd_id
                    self._insert_into_bucket(address)
                if address not in self.addrMan:
                    self.addrMan[address] = t
                self.address_known[sender_node_id][address] = rnd_id
                if self._is_terrible(t, address):
                    self.addrMan[address] = t - 5 * 60 * 60



        # get address call
        if envelope['get_address'] is True:
            addresses_to_send = dict()
            node_ids = self._closest_addresses(envelope['random_address'])
            for ii in node_ids:
                if ii == sender_node_id:
                    continue
                addresses_to_send[ii] = self.node_id_to_rnd[ii]
            answer_envelope['address_list'] = [addresses_to_send]
            answer_envelope['random_address'] = envelope['random_address']
        self.output_envelope = [answer_envelope]
        return answer_envelope

    def ask_neighbour_to_get_addresses(self, t):
        random_address = format(random.randint(0, 2 ** 32), '032b')
        output = []
        addresses_to_ask = self._closest_addresses(random_address)
        for ii in addresses_to_ask:
            envelope = self._get_empty_envelope(t, ii, random_address)
            envelope['get_address'] = True
            output.append(envelope)
        if len(addresses_to_ask) > 0:
            self.output_envelope = [envelope]
        return output

    def ask_node_for_self_lookup(self,t):
        output = []
        for ii in self._hard_coded_dns:
            envelope = self._get_empty_envelope(t, ii, self.rnd_id)
            envelope['get_address'] = True
            output.append(envelope)
        return output

    def ask_node_to_get_address(self,t, rnd_address, address_to_ask):
        envelope = self._get_empty_envelope(t, address_to_ask, rnd_address)
        envelope['get_address'] = True
        return envelope

    def buffer_to_send(self, addresses, neighbour):
        for address in addresses:
            if (address not in self.address_known[neighbour])\
                    or (self.addrMan[address] > self.address_known[neighbour][address]):
                timestamp = self.addrMan[address]
                self.address_buffer.append({address: timestamp})

    def interval_processes(self, t):
        self.output_envelope = []
        output = []
        if t - self.interval_timestamps['30min'] > 30*60:
            self.interval_timestamps['30min'] = t
            output = self.ask_neighbour_to_get_addresses(t)
        if len(self.outbound) > self.MAX_OUTBOUND_CONNECTIONS:
            envelope3 = self._delete_oldest_outbound_connection(t)
            if envelope3 is not None:
                self.output_envelope.append(envelope3)
                output.append(envelope3)
        if len(self.outbound) < self.MAX_OUTBOUND_CONNECTIONS:
            neighbours_to_connect = self.MAX_OUTBOUND_CONNECTIONS - len(self.outbound)
            for _ in range(neighbours_to_connect):
                envelope4 = self.update_outbound_connections(t)
                if envelope4 is not None:
                    self.output_envelope.append(envelope4)
                    output.append(envelope4)
                else:
                    break
        outdated_connections = self._outdated_connections(t)
        self.output_envelope.extend(outdated_connections)
        output.extend(outdated_connections)
        return output

    ############################
    # private functions
    ############################

    def _outdated_connections(self, t):
        envelopes = []
        for address, timestamp in self.outbound.items():
            if self.addrMan[address] + self.ADDRMAN_REPLACEMENT_HOURS * 60 * 60 < timestamp:
                envelope = self._get_empty_envelope(t, address)
                envelope['kill_connection'] = True
                envelopes.append(envelope)
        return envelopes

    def _delete_oldest_outbound_connection(self, t):
        oldest_outbound_node_address = min(self.outbound, key=self.outbound.get)
        # oldest_address = [key for key, value in self.outbound.items() if value == oldest_timestamp][0]
        return self._kill_connection(t, oldest_outbound_node_address)

    def _get_address_to_connect(self):
        if len(self.record_outgoing_conn['RandomNodes']) <= math.floor(self.MAX_OUTBOUND_CONNECTIONS * 0.5):
            #in this case, we need to choose u.a.r an entry from the table
            #for simplicity (but still correct) we choose an entry from another dict
            i = 0
            while True:
                i += 1
                if i >= 10:
                    break
                try_address = random.choice(list(self.node_id_to_rnd.keys()))
                if (try_address not in self.outbound and try_address not in self.record_outgoing_conn['RandomNodes']
                    and try_address not in self.record_outgoing_conn['RandomNodes'] and try_address != self.id):
                    break
            if i < 10:
                self.record_outgoing_conn['RandomNodes'].insert(0, try_address)
                return try_address
            else:
                #self._handle_record_out_conn()
                return None
        elif len(self.record_outgoing_conn['RandomNodes']) > math.floor(self.MAX_OUTBOUND_CONNECTIONS * 0.5):
            #in this case we take the first entry of the lookupBuf
            #the most recent discovered peers
            i = 0
            while True:
                i += 1
                if i >= 10:
                    break
                if len(self.lookupBuf) > 0:
                    try_address = list(self.lookupBuf.keys()).pop(0)
                    del self.lookupBuf[try_address]
                    if (try_address not in self.outbound and try_address not in self.record_outgoing_conn['lookupBuf']
                        and try_address not in self.record_outgoing_conn['RandomNodes'] and try_address != self.id):
                        break
                else:
                    # our buffer is empty, so return None to discover new peers
                    #self._handle_record_out_conn()
                    #peers with low id need to connect to DNS nodes
                    if self.id < len(self.DNS_rnd) + 5:
                        try_address = random.choice(list(set(self.DNS_rnd.keys()).difference(self.outbound)))
                        break
                    else:
                        return None
            if i < 10:
                self.record_outgoing_conn['lookupBuf'].insert(0, try_address)
                return try_address
            else:
                #self._handle_record_out_conn()
                return None
        else:
            return None

    def _kill_connection(self, t, address):
        envelope = self._get_empty_envelope(t, address)
        envelope['kill_connection'] = True
        return envelope

    def _initialize_outgoing_connections(self, t):
        if self._is_hard_coded_DNS is True:
            return None
        envelope = self._get_empty_envelope(t, random.choice(self._hard_coded_dns), format(random.randint(0, 2 ** 32), '032b'))
        envelope['get_address'] = True
        self.output_envelope = envelope
        return envelope

    def _is_terrible(self, t, address):
        if self.addrMan[address] >= t - 60:
            # never remove things tried in the last minute
            return False
        if self.addrMan[address] > t + 10 * 60:
            # came in a flying DeLorean
            return True
        if t - self.addrMan[address] > self.ADDRMAN_HORIZON_DAYS * 24 * 60 * 60:
            # not seen in recent history
            return True

        # could include number of attempts to connect

        return False

    def _get_empty_envelope(self, t, receiver, random_address_target=-1):
        return dict(sender={self.id: self.rnd_id}, receiver=receiver, timestamp=t, random_address=random_address_target, address_list=dict(), get_address=False, version=False,
                    connect_as_outbound=None, kill_connection=False, connection_killed=False)

    def _compare_bitwise(self, rnd_address):
        for i in range(self.P2P_BUCKET_SIZE):
            if rnd_address[i] != self.rnd_id[i]:
                return i
        return 15

    def _compute_bucket(self, rnd_address):
        return self._compare_bitwise(rnd_address)

    def _dns_to_bucket(self, node_id):
        bucket = self._compare_bitwise(self.DNS_rnd[node_id])
        self.bucket[bucket].insert(0, node_id)
        return

    def _closest_addresses(self, random_address):
        chosen_addresses = []
        bucket = self._compute_bucket(random_address)
        chosen_addresses = self.bucket[bucket]
        return chosen_addresses

    def _insert_into_bucket(self, node_id):
        rnd_address = self.node_id_to_rnd[node_id]
        bucket = self._compute_bucket(rnd_address)
        bucket_list = self.bucket[bucket]
        if node_id in bucket_list:
            return
        if len(bucket_list) >= 16:
            #check if the last entry in the bucket is online
            last_entry = bucket_list.pop()
            if last_entry in list(self.DG.nodes):
                self.bucket[bucket].insert(0, last_entry)
                del self.node_id_to_rnd[node_id]
            else:
                self.bucket[bucket].insert(0, node_id)
                del self.node_id_to_rnd[last_entry]
        else:
            #just insert
            self.bucket[bucket].insert(0, node_id)

    def _add_to_lookupBuf(self, sender_node_id, sender_rnd_id):
        while len(self.lookupBuf) >= 15:
            remove_key = list(self.lookupBuf.keys()).pop(0)
            del self.lookupBuf[remove_key]
        self.lookupBuf[sender_node_id] = sender_rnd_id

    def _handle_record_out_conn(self):
        copy = self.record_outgoing_conn['RandomNodes'].copy()
        for node in copy:
            if node not in self.outbound:
                self.record_outgoing_conn['RandomNodes'].remove(node)
        copy = self.record_outgoing_conn['lookupBuf'].copy()
        for node in copy:
            if node not in self.outbound:
                self.record_outgoing_conn['lookupBuf'].remove(node)
        return


############################
# static functions
############################
def _intersection_of_2_lists(a, b):
    return set(list(a)).intersection(set(b))