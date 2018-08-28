import os
import pandas as pd
import numpy as np
import random
import operator
from sklearn.cross_validation import train_test_split


class KnowledgeGraph:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.nodes = []  #nodes' id, shape=[10312]
        self.nodes_num = 0
        self.groups = [] #groups' id, shape=[39]
        self.groups_num = 0
        self.node_neighbors_dict = {} #dictionary, {node_id:neighbor_list}, list contain neighbors' id
        self.node_embedding_dict = {} #dictionary, {node_id:node_embedding}
        '''dataset divide'''
        self.training_nodes = []
        self.training_embed = []
        self.neighbors_embeds = []
        self.training_num = 0
        self.validation_nodes = []
        self.validation_embed = []
        self.validation_num = 0
        self.testing_ndoes = []
        self.testing_embed = []
        self.testing_num = 0
        self.index_in_epoch = 0
        self.epochs_completed = 0
        '''load data'''
        self.load_data()
        '''divide data set'''
        self.divide_dataset()
        

    def load_data(self):
        node_fileName = 'nodes.csv'
        group_fileName = 'groups.csv'
        node_file = pd.read_csv(os.path.join(self.data_dir, node_fileName), header=None)
        self.nodes = list(node_file[0])
        self.nodes_num = len(self.nodes)
        group_file = pd.read_csv(os.path.join(self.data_dir, group_fileName), header=None)
        self.groups = group_file[0]
        self.groups_num = len(self.groups)
        # print(self.nodes[10311])
        print('#number of nodes: {}'.format(self.nodes_num))        
        print('#number of groups: {}'.format(self.groups_num))
        edge_fileName = 'edges.csv'
        edge_file = pd.read_csv(os.path.join(self.data_dir, edge_fileName), header=None)
        cnt=0
        for i in range(len(edge_file[0])):
            node1, node2 = edge_file[0][i], edge_file[1][i]
            if self.node_neighbors_dict.get(node1,-1) == -1:  # node1 not exits in dict, then create set
                self.node_neighbors_dict[node1] = set([node2])
                cnt += 1
            else: # node1(key) already exits in dict, then add
                self.node_neighbors_dict[node1].add(node2)
            if self.node_neighbors_dict.get(node2,-1) == -1:
                self.node_neighbors_dict[node2] = set([node1])
                cnt += 1
            else:
                self.node_neighbors_dict[node2].add(node1)
        print('number of connected nodes: {}'.format(cnt))
        # print(self.node_neighbors_dict)
        
        '''get the biggest neighbor's number'''
        '''
        max_len=-
        for j in range(1,self.nodes_num+1):
            if self.node_neighbors_dict.get(j,-1) != -1:
                nei_len = len(self.node_neighbors_dict[j])
                max_len = max(max_len, nei_len)
        print(max_len)
        '''

        node_group_fileName = 'group-edges.csv'
        node_group_file = pd.read_csv(os.path.join(self.data_dir, node_group_fileName), header=None)
        # print(len(node_group_file))
        node_embedding_matrix = np.zeros((self.nodes_num,self.groups_num)) # start with 0,node_embedding_matrix[1][2]=1 means node 2 has label 3(that also means group 3)
        #cnt1 = 0
        for i in range(len(node_group_file[0])):
            node_embedding_matrix[node_group_file[0][i]-1][node_group_file[1][i]-1] = 1 
            #cnt1 += 1
        #print(cnt1)
        #for j in range(27,32):
            #print(node_embedding_matrix[j])
        self.node_embedding_dict = dict(zip(self.nodes, node_embedding_matrix[:]))
        #print(self.node_embedding_dict[28])


    def divide_dataset(self):
        '''
        self.training_num = round(self.nodes_num*0.8)
        self.training_nodes = self.nodes[:self.training_num]
        self.training_embed = np.array(operator.itemgetter(*self.training_nodes)(self.node_embedding_dict))
        '''
        #self.training_num = self.nodes_num
        self.training_nodes = self.nodes
        #self.training_embed = np.array(operator.itemgetter(*self.training_nodes)(self.node_embedding_dict))
        '''add nodes to make dataset balanced'''
        total_add = []
        for nod in self.training_nodes:
            label = np.argmax(self.node_embedding_dict[nod])
            if label==0:
                added = [nod]*9
            elif label==1 or label==2 or label==4 or label==5 or label==18 or label==23:
                added = [nod]
            elif label==3 or label==20 or label==24:
                added = [nod]*7
            elif label==6 or label==12:
                added = [nod]*2
            elif label==8 or label==13 or label==17 or label==19 or label==21 or label==22 or label==25:
                added = [nod]*4
            elif label==9 or label==10 or label==15 or label==16 or label==29 or label==31:
                added = [nod]*3
            elif label==11:
                added = [nod]*40
            elif label==14 or label==26 or label==30 or label==32:
                added = [nod]*17
            elif label==27:
                added = [nod]*13
            elif label==28:
                added = [nod]*8
            elif label==33 or label==34 or label==36:
                added = [nod]*26
            elif label==35:
                added = [nod]*11
            elif label==37:
                added = [nod]*59
            elif label==38:
                added = [nod]*201
            else: #label==7,not need to add
                added = []
            total_add.extend(added)
            #print('node{} add{}'.format(nod,len(total_add)))
        self.training_nodes.extend(total_add)
        self.training_num = len(self.training_nodes)
        print('#training num: {}'.format(self.training_num))
        self.training_nodes = np.array(self.training_nodes)
        self.training_embed = np.array(operator.itemgetter(*self.training_nodes)(self.node_embedding_dict))
        # Shuffle the data
        perm = np.arange(self.training_num)
        np.random.shuffle(perm)
        self.training_nodes = self.training_nodes[perm]
        self.training_embed = self.training_embed[perm]

        '''
        self.validation_num = round(self.nodes_num*0.1)
        self.validation_nodes = self.nodes[self.training_num : self.training_num + self.validation_num]
        self.testing_num = self.nodes_num - self.training_num - self.validation_num
        self.testing_nodes = self.nodes[self.training_num + self.validation_num :]
        print('#number of training,validation,testing nodes: %d %d %d,  total:%d'%(self.training_num,self.validation_num,self.testing_num,len(self.training_nodes)+len(self.validation_nodes)+len(self.testing_nodes)))
        '''
        '''split dataset'''
        '''
        self.training_nodes, self.testing_nodes, self.training_embed, self.testing_embed = train_test_split(self.training_nodes, self.training_embed, test_size=0.9995, random_state=0)
        self.training_num = len(self.training_nodes)
        self.testing_num = len(self.testing_nodes)
        print('training num:',self.training_num)
        print('testing num:',self.testing_num)
        #print(self.testing_nodes)
        #print(self.testing_embed)
        print(self.training_nodes)
        print(self.training_embed)
        '''
 
    '''only for training data'''
    def next_batch(self, batch_size):            
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.training_num:
            # Finish epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.training_num)
            np.random.shuffle(perm)
            self.training_nodes = self.training_nodes[perm]
            self.training_embed = self.training_embed[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.training_num
        end = self.index_in_epoch
        neighbors_embeds = [] # 2-dim, all neighbors' embedding concat
        neighbors_belong_ids = [] #1-dim, e.g[0,0,0,1,1] means the first three embeddings of neighbors_embeds belong of one node
        neighbors_num = []
        _id = 0
        for node in self.training_nodes[start:end]:
            neighbors_num.append(len(self.node_neighbors_dict[node]))
            for nei in self.node_neighbors_dict[node]:
                #if nei < self.training_num:
                neighbors_embeds.append(self.node_embedding_dict[nei])
                neighbors_belong_ids.append(_id)
            _id += 1        
        self.neighbors_embeds = neighbors_embeds
        return self.training_nodes[start:end], self.training_embed[start:end], self.neighbors_embeds, neighbors_belong_ids, neighbors_num


if __name__ == '__main__':
    kg =  KnowledgeGraph(data_dir='../data/')
    kg.index_in_epoch = 8249
    node,label,nei_emb,nei_ids,nei_num = kg.next_batch(4)
    print('node id: {}'.format(node))
    print('label: {}'.format(label))
    #print('nei_emb: {}'.format(nei_emb))
    #print('nei_ids: {}'.format(nei_ids))
    print('len(nei_emb): {} len(nei_ids): {}'.format(len(nei_emb), len(nei_ids)))
    print('nei_num: {}'.format(nei_num))
