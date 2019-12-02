#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:23:29 2019

@author: Yu Zhou
"""


import os
import sys

project_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"  # 将2, 3, 4, 5号GPU作为备选GPU

import torch
device = torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

import math
import numpy as np

from Utils.args import parse_args
from Utils.config import read_config
from Utils.dataset import Dataset, GensimData, Result, Name, numerate_dataset, form_gensim_docs

from Utils.utils import trace, pk, loadpk
from Utils.utils import ppl, draw_ppl, get_cohs, display_topics
from Utils.utils import batch_to_tensor , get_tw_list , get_cohs
#from Utils.scholar_utils import load_word_vectors, get_init_bg, load_covariates
#from Utils.gcn_utils import preprocess_features, preprocess_adj， masked_loss, masked_acc
from Utils.line_utils import makeDist, VoseAlias, get_topics_from_embs, distance
from Utils.line_utils import read_emb, generate_k_centroids
from Utils.utils import get_line_ge
from Utils.dec_utils import dec_pytorch
from Utils.mygcn_utils import *
from Utils.graph_utils import sample_from_matrix, build_geinput

from gensim.models.wrappers import LdaMallet
from Models.allModels.scholar import torchScholar
from collections import defaultdict
import networkx as nx
from node2vec import Node2Vec as NV

import subprocess

###############################################################################

def write_netfile(wtoi, data_set, word_file_path, idx_file_path):
    d = {}
    for doc in data_set:
        words = list(set(doc))
        for i in range(len(words)-1):
            for j in range( i+1 , len(words)):
                pair_1 = tuple((words[i] , words[j]))
                pair_2 = tuple((words[j] , words[i]))
                if pair_1 in d:
                    d[pair_1] += 1
                    d[pair_2] += 1
                else:
                    d[pair_1] = 1
                    d[pair_2] = 1
    trace('{} pair(edges) catched.'.format(len(d)), file=config.log_file)
    with open(word_file_path, 'w') as writer1 , open(idx_file_path, 'w') as writer2:
        for pair_set , value in d.items():
            if value<5:
                continue
            w1,w2 = pair_set
            writer1.write('{} {} {}\n'.format(w1,w2,value))
            writer2.write('{} {} {}\n'.format(wtoi[w1],wtoi[w2],value))
            #w.write('{} {} {}\n'.format(w2,w1,value))
        writer1.close()
        writer2.close()
        
def neg_init(array, v):
    rows, cols = array.shape
    for row in range(rows):
        for col in range(cols):
            array[row][col] = v
    return array
        
def build_twnp(assignments, centroids, nodes_i2w, word_emb, wtoi):
    num_topics = len(centroids) ; num_words = len(wtoi)
    trace('tw_np shape: {} x {} , non-zero value(word_emb) : {}'.format(num_topics, num_words, len(word_emb)), file=config.log_file)
    tw_np = np.zeros((num_topics, num_words))
    tw_np = neg_init(tw_np, -99)
    print('negatived np')
    
    for ass_idx, node_indices in assignments.items():
        for node_idx in node_indices:
            word = nodes_i2w[node_idx] 
            word_idx = wtoi[word]
                
            dist = distance(centroids[ass_idx], word_emb[word])
            tw_np[ass_idx][word_idx] = -dist
    return tw_np
                        
    
def inverse_d(d):
    new_d = {}
    for key, value in d.items():
        new_d[value] = key
    return new_d
###############################################################################
    
   

            
    
def run_cmd(config, model, graph_file, emb_file, geinput=None):
    while True:
        if model == 'LINEs':
            git_path = os.path.abspath(os.path.dirname(project_path))
            line_path = os.path.abspath(os.path.join(git_path, 'LINE/linux/line'))
            get_line_ge(line_path, graph_file, emb_file, config.h_dim)  
            
        elif model == 'PyGCN':
            graph = geinput
            node2vec = NV(graph, dimensions=config.h_dim, walk_length=30, num_walks=200, workers=4)
            g_model = node2vec.fit(window=10, min_count=1, batch_words=4)
            g_model.wv.save_word2vec_format(emb_file)

        if os.path.isfile(emb_file):
            break
        else:
            print('{} emb_file writing failed.'.format(model))
        
    return

def nodes2array(nodes):
    array = np.zeros((len(nodes), len(nodes[1])), dtype=np.float32)
    for i in range(len(nodes)):
        array[i] = nodes[i].reshape(1,-1)
    return array

def array2assi(array):
    rows, cols = array.reshape(-1,1).shape
    d = {}
    for row in range(rows):
        if array[row] in d:
            d[array[row]].append(row)
        else:
            d[array[row]] = [row]
    return d

def array2list(array):
    rows, cols = array.shape
    l = []
    for row in range(rows):
        l.append(array[row])
    return l

def cluster_ge(t, nodes, nodes_i2w, word_emb, emb_file, wtoi, itow, init_c=None):
    if config.cluster == 'kmeans':
        assignments, centroids, nodes_i2w, word_emb = get_topics_from_embs(nodes, nodes_i2w, word_emb, num_topics=config.z_dim, init_c=widget)
#        from sklearn.cluster import KMeans
#        sk_kmeans = KMeans(n_clusters=config.z_dim)
#        data = nodes2array(nodes)
#        sk_kmeans.fit(data)
#        assignments = {}
#        for node_idx, cluster_idx in enumerate(sk_kmeans.predict(data)):
#            if cluster_idx in assignments:
#                assignments[cluster_idx].append(node_idx)
#            else:
#                assignments[cluster_idx] = [node_idx]
#        centroids = []
#        for item in sk_kmeans.cluster_centers_:
#            centroids.append(item)
        

    elif config.cluster == 'dec':
        #init_centroids = generate_k_centroids(nodes, config.z_dim, m=config.init_way, init_c=init_c)
        #data_c = np.array(init_centroids)
        data_x = nodes2array(nodes)
        centroids_np, assignments = dec_pytorch(config, data_x, init_c=init_c)
        #print(centroids_np.shape, assignments.shape)
        assignments = array2assi(assignments)
        centroids = array2list(centroids_np)
#        for k,v in assignments.items():
#            print(k, len(v))
#    print('wtoi size: {}'.format(len(wtoi)))
#    for m, key in enumerate(wtoi.keys()):
#        if m>10:
#            break
#        else:
#            print(key, wtoi[key])
    tw_np = build_twnp(assignments, centroids, nodes_i2w, word_emb, wtoi)
    print('tw_np')
    tw_tensor = torch.from_numpy(tw_np)
    print('tw_tensor')
    tw_list = get_tw_list(tw_tensor, itow, k=10)
    print('tw_list')
    cohs = get_cohs(tw_list)
    print('cohs')
    avg_coh = sum(cohs)/len(cohs)
#        p = ppl(test_data, tw_tensor)
#        print('ppl')
    TWmatrix.append(tw_np)
    TWlist.append(tw_list)
    COHs.append(cohs)        
    avg_COHs.append(avg_coh) 
#        PPLs.append(p)
    
    seg = '---------- topics of time {} ----------'.format(t)
    display_topics(tw_list=tw_list, cohs=cohs,
                   head='topics', seg=seg, 
                   file = config.topic_file)
    return 'null'

def write_topicmap(topic_file, output_dir, T):
    print(topic_file)
    for tj in range(T):
        reader_dir = os.path.join(output_dir, 'result_{}'.format(tj))
        if not os.path.isdir(reader_dir):
            continue
        reader = os.path.join(reader_dir, 'lda_summary_final.txt')
        tw_list = []
        with open(reader, 'r') as r:
            for i,line in enumerate(r):
                if i%2 == 0:
                    continue
                words = line.strip().split()
                words = words[:10]
                while len(words) != 10:
                    words.append('null')
                tw_list.append(words)
            r.close()
    cohs = get_cohs(tw_list)
    seg = '---------- topics of time {} ----------'.format(t)
    display_topics(tw_list=tw_list, cohs=cohs,
                       head='topics', seg=seg, 
                       file = topic_file)
    return
    

def main(t, T, train_data, widget, whole_wtoi, whole_itow):
    
    # Create dict of distribution when opening file
#    edge_dist_dict, node_dist_dict, weights, nodedegrees, maxindex = makeDist(
#        config.graph_path, config.negativepower)
#    edges_alias_sampler = VoseAlias(edge_dist_dict)
#    nodes_alias_sampler = VoseAlias(node_dist_dict)     
    
    # choose graph type
    #model = choose_graph(config)
    model = config.graph
    print('model: ', model)
    # build input for graph embedding
    if model == 'TopicMap':
        graph_file= os.path.join(config.output_path, 'temp_graph_file_{}.txt'.format(t))
    elif model == 'LINEs':
        graph_file = 'temp_graph_file_{}.txt'.format(t)
    elif model == 'PyGCN':
        graph_file = 'temp_graph_file_{}.txt'.format(t)
    elif model == 'MyGCN':
        graph_file = 'temp_graph_file_{}.txt'.format(t)
    
        
    geinput = build_geinput(model, train_data, whole_wtoi, graph_file)
    
    if model == 'TopicMap':
        centroids = 0
        git_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
        topicmap_path = os.path.join(git_path, 'topicmapping/bin/topicmap')
        emb_dir = os.path.join(config.output_path, 'result_{}'.format(t))
        cmd = '{} -f {} -t 10 -o {}'.format(
                topicmap_path, graph_file, emb_dir)
        print(cmd)
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #centroids = subp.stdout.readlines() 
        write_topicmap(config.topic_file, config.output_path, T)

    elif model == 'LINEs' or model == 'PyGCN':
        # get embedding
        emb_file = 'temp_emb_file.txt'
        run_cmd(config, model, graph_file, emb_file, geinput)
        nodes, nodes_i2w, word_emb = read_emb(emb_file)
        # clustering
        if FLAG:
            widget = widget
        else:
            widget = None
        centroids = cluster_ge(t, nodes, nodes_i2w, word_emb, emb_file, wtoi=whole_wtoi, itow=whole_itow, init_c=widget)
        os.remove(emb_file)
        
    elif model == 'MyGCN':
        A_matrix, X_matrix, cur_wtoi, cur_itow = geinput
        if FLAG:
            print('new shape of X_matrix is: {}'.format(X_matrix.shape))
            init_weight_11 = np.random.random((len(cur_wtoi), config.h_dim))
            a=0; b=0
            for word, cur_w_i in cur_wtoi.items():
                memory_w_v = MemoryVD[whole_wtoi[word]]
                if np.all(memory_w_v==0):
                    b+=1
                else:
                    a+=1
                    init_weight_11[cur_w_i] = memory_w_v
            s = "inherit {} word embeddings and random {}".format(a,b)
            trace(s, file=config.log_file)
            w12, w21 = widget
            init_weight = (init_weight_11, w12, w21)
        else:
#            init_weight = np.random.random((len(whole_wtoi), config.h_dim))
#            print(init_weight)
            init_weight = None
            
        N_ , F_ = X_matrix.shape ; D_ = config.h_dim 
        print(N_, F_, D_)
            
        model_gcn = GCN_model(config, n_dim=N_, d_dim=D_, f_dim=F_,
                              init_weight_np=init_weight)
#        print(model_gcn.weight_11.data)
        model_gcn.to(config.device)
        optimizer = torch.optim.Adam(model_gcn.parameters(), lr=config.lr)
        model_gcn.train()
        for epoch in range(config.epochs):
            b = 5 ; k = math.ceil(N_/b) #; print('k: ',k)
            for  batch in range(2*b):
                cur_a, chosen_idx = sample_from_matrix(A_matrix, k=k)
                cur_x, _ = sample_from_matrix(X_matrix, k=k, chosen_idx=chosen_idx)
                optimizer.zero_grad()
                cur_inputs = (torch.tensor(cur_a).to(config.device), torch.tensor(cur_x).to(config.device))
#                print(cur_inputs[0].dtype)
#                print(model_gcn.weight_11.data.dtype)
                rec, loss = model_gcn(cur_inputs)
                loss.backward()
                optimizer.step()
            cur_inputs = (torch.tensor(A_matrix).to(config.device), torch.tensor(X_matrix).to(config.device))
            optimizer.zero_grad()
            rec, loss = model_gcn(cur_inputs)
            loss.backward()
            optimizer.step()
            if epoch%10==0:
                s = "epoch:{}, loss:{}".format(epoch, loss)
                trace(s, file=config.log_file)
                
        # update memory VD (node/word embedding)
        nd_matrix, w12, w21 = model_gcn.get_widget()
#        print('nd_matrix\n', nd_matrix) 
        c = 0 ; d = 0
        for cur_w_i, word in cur_itow.items():
            memory_w_i = whole_wtoi[word]
            cur_v = nd_matrix[cur_w_i]
            if np.all(cur_v==0):
                c += 1
            else:
                MemoryVD[memory_w_i] = cur_v
                d += 1
        trace('update {}/{} words from cur to memory'.format(d, c))
            
            
        # write emb_file
        emb_file = 'temp_emb_file.txt'
        with open(emb_file, 'w') as writer:
            writer.write('{} {}'.format(N_, config.h_dim))
            for row_i, row in enumerate(nd_matrix):
                word = cur_itow[row_i]
                vector_str = [str(item) for item in row]
                s = "{} {}\n".format(word, ' '.join(vector_str))
                writer.write(s)  
            
        # clustering
        nodes, nodes_i2w, word_emb = read_emb(emb_file)
        centroids = cluster_ge(t, nodes, nodes_i2w, word_emb, emb_file, wtoi=whole_wtoi, itow=whole_itow, init_c=widget)
        return (w12, w21)
    #os.remove(graph_file)
    return centroids
    
    

if __name__ == '__main__':
    
    # configuration
    args, parser = parse_args()
    config_file = 'config/tryconfig.ini'
    global config
    config = read_config(args, parser, config_file)
    s = 'Start running ges.py \n {}\n'.format(str(config))
    trace(s,file=config.log_file, write='w')
    global T
    train_set = Dataset(config)
    T = train_set.T()
    global TWmatrix, TWlist, COHs, PPLs, avg_COHs
    TWmatrix = [] ; TWlist = []; COHs = []; PPLs = []
    avg_COHs = []
    
    name = Name(flag=config.flag, config=config,
                model_name=config.graph, data_name=config.train_file, time_slices=train_set.time_slices[:-1])
    result = Result(info=name, TWmatrix=TWmatrix, itow=train_set.itow,
                    twlist=TWlist, COHs=COHs, PPLs=PPLs)
    result_file = os.path.join(config.output_path, 'result')

    if config.graph == 'MyGCN':
        global MemoryVD, V
        V = len(train_set.cur_counter)
        MemoryVD = np.zeros((V,config.h_dim))
    
    try:
        global test_data
        test_data = form_gensim_docs(train_set.test(), train_set.wtoi)
        global FLAG
        FLAG = False
        widget = None
        for t, train_data in enumerate(train_set):
            trace('time {}/{}'.format(t,T), file=config.log_file)
            widget = main(t, T, train_data, widget, train_set.wtoi, train_set.itow)
            FLAG = config.continuous
                
        result.TWmatrix = TWmatrix
        result.twlist = TWlist
        result.COHs = COHs
        result.PPLs = PPLs
        result.tag = 'complete'
        pk(result_file, result)
        
#        topic_evos, cohs_evos = result.get_evos()
#        i = 0
#        for cohs_evo, topic_evo in zip(cohs_evos, topic_evos):
#            seg = '---------- topics {} evolution ----------'.format(i)
#            display_topics(tw_list=topic_evo, cohs=cohs_evo,
#                                   head='time', seg=seg, 
#                                   file = config.topic_evo_file)
#            i += 1
        
    except KeyboardInterrupt:
        result.TWmatrix = TWmatrix
        result.twlist = TWlist
        result.COHs = COHs
        result.PPLs = PPLs
        result.tag = 'incomplete'
        pk(result_file, result)        
    
#    vocab_path = os.path.join(config.data_path, config.vocab_filename)     
#    embedding_file = os.path.join(config.data_path, config.emb_filename)
#
#    if config.ge_input:
#        # create or load emb_file and vocab
#        if os.path.isfile(embedding_file):
#            print('embedding file already exist, please continue.')
#        else:       
#            train_set = Dataset(config)        
#            net_path = os.path.join(config.data_path, '{}_{}_{}'.format(config.filename, config.unit, config.time_period))
#            if not os.path.isdir(net_path):
#                os.makedirs(net_path)        
#            vocab_path = os.path.join(net_path, 'vocab.pkl')
#            vocab = train_set.wtoi        
#            for t,train_data in enumerate(train_set):
#                word_netfile_path = os.path.join(net_path, 'word_net_{}.txt'.format(t))
#                idx_netfile_path = os.path.join(net_path, 'idx_net_{}.txt'.format(t))
#                write_netfile(vocab, train_data, word_netfile_path, idx_netfile_path)    
#            pk(vocab_path, vocab)   
#            itow = inverse_d(vocab)
#    if config.ge_cluster:
#        
#        global test_topic_file
#        test_topic_file = 'test_topics_from_line.txt'  
#    
#        vocab = loadpk(vocab_path)
#        itow = inverse_d(vocab)
#        emb_topic_words(embedding_file, vocab, itow, config.z_dim)
    
    #main(config)
            

    