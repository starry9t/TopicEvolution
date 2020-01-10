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

import torch
import numpy as np

from Utils.args import parse_args
from Utils.config import read_config
from Utils.dataset import Dataset, GensimData, Result, Name

from Utils.utils import trace, pk#, loadpk
from Utils.utils import ppl, draw_ppl, get_cohs, display_topics, get_tw_list

from gensim.models.wrappers import DtmModel


#def get_topic_string(sample_topic):
#    sample_topic = sample_topic[0][1]
#    res = ""
#    for i, token in enumerate(sample_topic.strip().split('\"')):
#        if i%2 == 1:
#            res += "{} ".format(token)
#    return res

def get_topic_np(sample_topic, num_topics, wtoi):
    num_words = len(sample_topic[0])
    res = np.zeros((num_topics, num_words))
    for topic_idx in range(num_topics):
        t_tuples = sample_topic[topic_idx]
        for t_tuple in t_tuples:
            word_idx = wtoi[t_tuple[1]]
            res[topic_idx][word_idx] = t_tuple[0]
        #probs = [t_tuple[0] for t_tuple in t_tuples]
        #res[topic_idx] = probs 
    return res



def get_dt_tensor(dt_list):
    topic_prob = [item[1] for item in dt_list]
    t = torch.from_numpy(np.array(topic_prob))
    return t
    

def main():
    trace('---train topics---', config.log_file)
    model = DtmModel(dtm_path, corpus=gensim_data.corpus,
                     id2word=gensim_data.dictionary,
                     time_slices=train_set.time_slices[:-1],
                     num_topics=config.z_dim,
                     lda_sequence_min_iter=50,
                     lda_sequence_max_iter=config.epochs)
    trace('---model trained---', config.log_file)
    #
    sample_topic = model.dtm_coherence(time=0, num_words=10)
    print('sample topic is like: {}'.format(' '.join(sample_topic[0])), config.log_file)
    
    #
    tw_nps = model.show_topics(num_topics=config.z_dim, times=-1,
                              num_words=train_set.vocab_size(),
                              formatted=False)

    for t in range(T):
        # topics in time t
        tw_np = tw_nps[t*config.z_dim : (t+1)*config.z_dim]
        
        tw_np = get_topic_np(tw_np, config.z_dim, gensim_data.dictionary.token2id)
        tw_tensor = torch.from_numpy(tw_np)
        tw_list_t = get_tw_list(tw_tensor, gensim_data.dictionary)
    
        # coh
        cohs_t = get_cohs(tw_list_t)
        p = ppl(gensim_data.test, tw_tensor)
    
        TWmatrix.append(tw_np)
        TWlist.append(tw_list_t)
        COHs.append(cohs_t)
        PPLs.append(p)
        
        avg_COHs.append((sum(cohs_t)/len(cohs_t)))
        
        seg = '---------- topics in time {}/{} ----------'.format(t+1, T)
        display_topics(tw_list=tw_list_t, cohs=cohs_t,
                       head='topics', seg=seg, 
                       file = config.topic_file)
        trace('topic result(coherence) written.', file=config.log_file)
        
    p_file = os.path.join(config.output_path, 'ppl.jpg')
    draw_ppl(PPLs, title='perplexities over time', file=p_file)
    a_file = os.path.join(config.output_path, 'avg_coh.jpg')
    draw_ppl(avg_COHs, title='avg coherence over time', file=a_file)
    
    
        # ppl
#        dts = list(model.load_document_topics())
#        dt_tensor = torch.zeros(len(gensim_data.corpus),config.z_dim)
#        d = 0
#        for dt in dts:
#            dt_tensor[d] = get_dt_tensor(dt)
#        p = ppl(gensim_data.corpus, dt_tensor, tw_tensor)
#        #cohs = ppl(gensim_data.corpus, dt_tensor, tw_tensor)
#        PPLs.append(p)
    #    seg = '---------- topics in time {}/{} ----------'.format(t+1, T)
    #    display_topics(tw_list=tw_list_t, cohs=None, coh_name='(ppl: {})'.format('%.4f' % p),
    #                   head='topics', seg=seg, 
    #                   file = config.topic_file)
    
    


    
    

if __name__ == '__main__':
    
    # configuration
    args, parser = parse_args()
    config_file = 'config/tryconfig.ini'
    global config
    config = read_config(args, parser, config_file)

    s = 'Start running m_DTM.py \n {}\n'.format(str(config))
    trace(s,file=config.log_file, write='w')
    
    global dtm_path
    dtm_path = os.path.join(project_path, 'dtm/dtm/main')

    # make dataset
    train_set = Dataset(config)
    global T
    T = train_set.T()
    
    global TWmatrix, TWlist, COHs, PPLs, avg_COHs
    TWmatrix = [] ; TWlist = []; COHs = []; PPLs = []
    avg_COHs = []
    
    n = train_set.time_slices[-1]
    train_data = [train_set.docs[idx] for idx in train_set.sorted_idx[:-n]]
    test_data = [train_set.docs[idx] for idx in train_set.sorted_idx[-n:]]
    gensim_data = GensimData(docs=train_data, test=test_data)
    
    name = Name(flag=config.flag, config=config,
                model_name='DTM', data_name=config.train_file, time_slices=train_set.time_slices[:-1])
    result = Result(info=name, TWmatrix=TWmatrix, itow=gensim_data.dictionary,
                    twlist=TWlist, COHs=COHs, PPLs=PPLs)
    result_file = os.path.join(config.output_path, 'result')
    
    try:
    #
        main()
        
        result.TWmatrix = TWmatrix
        result.twlist = TWlist
        result.COHs = COHs
        result.PPLs = PPLs
        result.tag = 'complete'
        pk(result_file, result)
        
    except KeyboardInterrupt:
        #result = Result(info=name, num_time=T, tws=TWs, COHs=COHs, PPLs=PPLs)
        result.TWmatrix = TWmatrix
        result.twlist = TWlist
        result.COHs = COHs
        result.PPLs = PPLs
        result.tag = 'incomplete'
        pk(result_file, result)
        
   

    
    ###


    
    