#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:11:35 2019

@author: Yu Zhou
"""

import os
import re
import sys
project_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_path)
git_path = os.path.abspath(os.path.dirname(project_path))
glove_path = os.path.abspath(os.path.join(git_path, 'glove'))
import argparse
from Utils.utils import loadpk, draw_tab, display_topics
from Utils.drawers import multi_lines
import numpy as np
import random 

import matplotlib
import matplotlib.pyplot as plt

key_words = ['vector', 'machine', 'unsupervised', 'validation', 'machines', 'supervised', 'word', 'deep', 'overfitting', 'sentiment', 'descent', 'clustering', 'networks', 'embedding', 'neural', 'computer vision', 'natural', 'cross', 'bayesian', 'regression', 'language', 'classification', 'support', 'learning', 'computational', 'robotics', 'decision', 'adversarial', 'gradient', 'association', 'trees', 'encoder', 'processing', 'algorithm', 'reinforcement', 'database', 'parsing']
key_words = set(key_words)

###############################################################################
def get_dirnames(odirs):
    dirnames = []
    #odirs = args.odirs.strip().split(' ')
    for odir in odirs:
        dirname = os.path.join(project_path, odir)
        dirnames.append(dirname)
    return dirnames

def select_outputs(flag, targets):
    if targets == None:
        return True
    tokens = flag.strip().split('_')
    if set(targets).issubset(set(tokens)):
        return True
    else:
        return False

def get_all_results(dirnames, target=None, result_tags=['complete']):
    res = []
    for dirname in dirnames:
        a = os.walk(dirname)
        for root, output_dirs, files in a:
            # dirs
            for output_dir in output_dirs:
                if not select_outputs(output_dir, target):
                    continue
                output_path = os.path.join(root, output_dir)
                result_path = os.path.join(output_path, 'result')
                if os.path.isfile(result_path):
                    result = loadpk(result_path)
                    try:                        
                        if result.tag in result_tags:
                            res.append((output_path,result)) 
                        else:
                            pass
                    except:
                        pass
                        #print('result has no attribute tag.')
                else:
                    #s = 'there is no result file in {}'.format(output_path)
                    #print(s)
                    pass
            break
    print('get {} results as below.'.format(len(res)))
    for dirname, _ in res:
        print(dirname)
    print('  ')
    return res

def get_results(odirs, targets, tags):
    dirnames = get_dirnames(odirs)    
    # all
    all_results = get_all_results(dirnames=dirnames, target=targets, result_tags=tags)
    return all_results
###############################################################################

def compare_best_topic(all_result, time=None):
    head = 'best_topic'
    row_names = []
    list_of_list = []
    last_col = []
    for (name, result) in all_result:
        row_names.append(name)
        best_topic, m_coh = result.show_best_topic(time)
        last_col.append(m_coh)
        list_of_list.append(best_topic)
    #m_idx = last_col.index(max(last_col))
    #last_col[m_idx] = '{}(max)'.format(last_col[m_idx])
        
    num_col = len(list_of_list[-1])
#    if num_col == 0:
#        return None
    col_names = ['word_{}'.format(i) for i in range(num_col)]
    last_col_name = 'coh'
    
    if time is not None:
        seg = 'best topics in time {}'.format(time)
    else:
        seg = 'best topics for all models'
        
    # sorted list_of_list last_col row_names
    array = np.array(last_col)
    sorted_idx = list(np.argsort(-array))
    
    list_of_list = [list_of_list[i] for i in sorted_idx]
    last_col = [last_col[i] for i in sorted_idx]
    row_names = [row_names[i] for i in sorted_idx]

    return list_of_list, last_col, last_col_name, head, row_names, col_names, seg

def compare_topic_cohs(args):
    all_results = get_results(args.odirs, args.target, args.tags)
    res = compare_best_topic(all_results)
    list_of_list, last_col, last_col_name, head, row_names, col_names, seg = res
    a_file = os.path.join(project_path, 'stat/compare_best_topic.txt')
    draw_tab(list_of_list, last_col, last_col_name, head, row_names, col_names, seg, a_file)
    
    base_targets = args.target
    # CL year
    s_target = base_targets + ['CL']
    all_results = get_results(args.odirs, s_target, args.tags) ; flag=True ; t=0
    while flag == True:
        try:
            res = compare_best_topic(all_results, t)
            list_of_list, last_col, last_col_name, head, row_names, col_names, seg = res
            a_file = os.path.join(project_path, 'stat/compare_best_topic_{}.txt'.format(s_target))
            draw_tab(list_of_list, last_col, last_col_name, head, row_names, col_names, seg, a_file) 
            t += 1
        except:
            flag == False
            break
        
    # LG year
    s_target = base_targets + ['LG']
    all_results = get_results(args.odirs, s_target, args.tags) ; flag=True ; t=0
    while flag == True:
        try:
            res = compare_best_topic(all_results, t)
            list_of_list, last_col, last_col_name, head, row_names, col_names, seg = res
            a_file = os.path.join(project_path, 'stat/compare_best_topic_{}.txt'.format(s_target))
            draw_tab(list_of_list, last_col, last_col_name, head, row_names, col_names, seg, a_file) 
            t += 1
        except:
            flag == False
            break
 
###############################################################################
       



###############################################################################

def load_word_embedding(file, emb_dim):
    if not os.path.isfile(file):
        print('embedding file {} does not exist!'.format(file))
        exit(0)
    embeddings = {}
    with open(file, 'r') as reader:
        for line in reader:
            tokens = line.strip().split(' ')
            if len(tokens) == emb_dim+1:
                word = tokens[0]
                embeddings[word] = np.array([float(item) for item in tokens[1:]])
        #print('{} words embedding loaded')
    return embeddings

def get_word_embedding(word, embeddings_d, emb_dim=100):
    try:
        emb = embeddings_d[word]
        return emb
    except:
        return np.random.random((emb_dim))

def np_similarity(np_a, np_b):
    return np.sqrt(np.sum(np.square(np_a-np_b)))

#def word_topic_sim(keyword, topic):
#    keyword_emb = get_word_embedding(keyword, embeddings, args.emb_dim)
#    sims = []
#    for word in topic:
#        word_emb = get_word_embedding(word, embeddings, args.emb_dim)
#        sim = np_similarity(keyword_emb, word_emb)
#        sims.append(sim)
#    return sum(sims)/len(sims)

def get_sim(word_a, word_b):
    emb_a = get_word_embedding(word_a, embeddings, args.emb_dim)
    emb_b = get_word_embedding(word_b, embeddings, args.emb_dim)
    res = np_similarity(emb_a, emb_b)
    return res

def get_words_sim(words_a, words_b):
    n_a = len(words_a) ; n_b = len(words_b)
    sim = 0.0
    for i_word_a in words_a:
        for i_word_b in words_b:
            sim_i = get_sim(i_word_a, i_word_b)
            sim += sim_i
    sim = sim/(n_a*n_b)
    return sim

def analize(result, word):
    #w_prob = result.get_word_prob(word)
    pass



def compare_most_similar_topic(all_result, time=None):
    pass

def get_similarities(result, keywords):
    x_axis = range(len(result.info.time_slices))
    
    y_values = []
    for tw_list in result.twlist:
        # time t
        sims_t = []
        for words in tw_list:
            topic_t_sim = get_words_sim(words, keywords)
            sims_t.append(topic_t_sim)
        sim_t = max(sims_t)
        y_values.append(sim_t)
    return (x_axis, y_values)

def compare_topic_sims(keywords=['method']):
    all_results = get_results(args.odirs, args.target, args.tags)
    items = []
    for name,result in all_results:
        # item = (time_nums, values)  for pyplot
        item = get_similarities(result, keywords)
        items.append((item, name))
    ##
    a_file = os.path.join(project_path, 'stat/compare_sim_to_{}.png'.format(keywords))
    multi_lines(items, a_file)
###############################################################################
    
def play_topic(args):
    results = get_results(odirs=args.odirs, targets=args.target, tags=args.tags)
    _, result = results[0]
    result.get_topic_of(topic_name='alrorithm')
    return

def add_coh(args):
    results = get_results(odirs=args.odirs, targets=args.target, tags=args.tags)
    for output_path, result in results:   
        t_file = os.path.join(output_path, 'topic_addc.txt')
        result.add_cohs(file=t_file)
        
def rewrite_topic(args):
    dirnames = get_dirnames(args.odirs)
    for dirname in dirnames:
        a = os.walk(dirname)
        for root, output_dirs, files in a:
            # dirs
            for output_dir in output_dirs:
                if not select_outputs(output_dir, args.target):
                    continue
                output_path = os.path.join(root, output_dir)
                result_path = os.path.join(output_path, 'result')
                if os.path.isfile(result_path):
                    result = loadpk(result_path)
                    a = 1
                    try:                        
                        if result.tag in args.tags:
                            pass
                            #evo_file = os.path.join(output_path, 'topics_evo.txt') ; print(evo_file)
                    except:
                        continue
                    
                    retopic_file = os.path.join(output_path, 'topics_.txt')
                    i = 0
                    for tw_list, cohs in zip(result.twlist, result.COHs):
                        seg = '---------- topics in time {}  ----------'.format(i)
                        display_topics(tw_list=tw_list, cohs=cohs,
                                               head='topic', seg=seg, 
                                               file = retopic_file)
                        i += 1
                else:
                    pass
            break
        
def rename_files(args):
    dirnames = get_dirnames(args.odirs)
    for dirname in dirnames:
        a = os.walk(dirname)
        for root, output_dirs, files in a:
            # dirs
            for output_dir in output_dirs:
                if not select_outputs(output_dir, args.target):
                    continue
                output_path = os.path.join(root, output_dir)
                topic_path = os.path.join(output_path, 'topics.txt')
                if os.path.isfile(topic_path):
                    flag = output_dir
                    new_topic_path = os.path.join(output_path, 'topics_{}.txt'.format(flag))
                    os.rename(topic_path,new_topic_path) 
                else:
                    pass
            break
   
def write_evos(all_results):
    for o_path, result in all_results:
        o_dir = o_path.split('/')[-1]
        evo_file = os.path.join(o_path, 'topics_evo_{}.txt'.format(o_dir))
        topic_evos, cohs_evos = result.get_evos()
        i = 0
        for cohs_evo, topic_evo in zip(cohs_evos, topic_evos):
            seg = '---------- evolution of topic {}  ----------'.format(i)
            display_topics(tw_list=topic_evo, cohs=cohs_evo,
                                   head='time', seg=seg, 
                                   file = evo_file)
            i += 1

     
###############################################################################
   
def draw_topic(pic, topic, t_color):
    num_words = len(topic)
    sum_x = 0.0 ; sum_y = 0.0
    for word in topic:
        x = coordinates[word][0] ; y = coordinates[word][1]
        pic.text(x, y, word,color=t_color)
        sum_x += x ; sum_y += y
    pic.plot(sum_x/num_words, sum_y/num_words, '*', color=t_color)
    #return sum_x/num_words, sum_y/num_words#

def draw_topics(topics):
    ordered_colors = ['green', 'blue']
    fig = plt.figure()
    pic = fig.add_subplot(111)
    #pic.spines['left'].set_color('none')
    pic.set_xlim(0.0,1.0)
    pic.set_ylim(0.0,1.0)
    pic.set_xticks([])
    pic.set_yticks([])
    for t_i, topic in enumerate(topics):
        draw_topic(pic, topic, ordered_colors[t_i])
        #pic.plot(c_x, c_y, '*', color=)
    fig.show()
    return
        
        
        
def draw_evo(topic_evo):
    '''
    topic_evo : list of word lists, e.g. [[surf, internet, online],[online, post, internet]...]
    '''
    return
    
    
if __name__ == "__main__":

    global args
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')
    
    parser.add_argument('--odirs', nargs='+', default=['output'], help="folders that contain folders of results")
    parser.add_argument('--target', nargs='+', default=['t20'], help="atrributes that contained in flag")
    parser.add_argument('--tags', nargs='+', default=['complete'], help="just seperate arguments")
    parser.add_argument('--sp1', default=1, help="just seperate arguments")
    parser.add_argument('--sp2', default=1, help="just seperate arguments")
    parser.add_argument('-evo', default=True, action='store_true', help="write best topics txt, default %default")
    
    
    parser_bt = subparsers.add_parser('bt', help='write best topics txt')
    parser_bt.add_argument('-bt', default=True, action='store_true', help="write best topics txt, default %default")
    
    parser_bt = subparsers.add_parser('sim', help='write best topics txt')
    parser_bt.add_argument('-sim', default=True, action='store_true', help="write best topics txt, default %default")
    parser.add_argument('--emb_dim', type=int, choices= [50,100,200,300], default=100, help='dimension of word embedding')
    parser.add_argument('--keywords', nargs='+', default=['method', 'algorithm'], help="just seperate arguments")
    
    #parser_bt = subparsers.add_parser('evo', help='write best topics txt')
    
    
    parser_bt = subparsers.add_parser('try', help='write best topics txt')
    parser_bt.add_argument('-try', default=True, action='store_true', help="write best topics txt, default %default")
    
    parser_addc = subparsers.add_parser('addc', help='supplement coherences to results')
    parser_addc.add_argument('-addc', default=True, action='store_true', help="write best topics txt, default %default")
    parser_addc.add_argument('-t_file', default=None, help="write best topics txt, default %default")
    
    parser_addc = subparsers.add_parser('recoh', help='supplement coherences to results')
    parser_addc.add_argument('-recoh', default=True, action='store_true', help="write best topics txt, default %default")

    parser_addc = subparsers.add_parser('rn', help='supplement coherences to results')
    parser_addc.add_argument('-rn', default=True, action='store_true', help="write best topics txt, default %default")
    
    args = parser.parse_args()
    
    all_results = get_results(args.odirs, args.target, args.tags)
    # o_path, result = all_results[i]
    if not all_results:
        print('sorry, no result of ({}) \nretrieved from \n{}'.format(args.tags, args.odirs))
        exit()
    
    if args.evo:
        print('prepare to write evolution files...')
        write_evos(all_results)        

    exit()
    if hasattr(args, 'addc'):
        add_coh(args)

    if hasattr(args, 'try'):
        play_topic(args)

    if hasattr(args, 'bt'):
        compare_topic_cohs(args)
        #compare_topic_cohs()
    if hasattr(args, 'sim'):
        # load word embedding (Dict)
        global embeddings
        embeddings = ''
        emb_file = os.path.join(glove_path, 'glove.6B.{}d.txt'.format(args.emb_dim))
        embeddings = load_word_embedding(emb_file, args.emb_dim)
        compare_topic_sims(keywords=args.keywords)
        
    if hasattr(args, 'recoh'):
        rewrite_topic(args)
    

        
    if hasattr(args, 'rn'):
        print('rename topics files and evo files')
        rename_files(args)
    
    
'''
    topics = [['dog', 'cat', 'rabbit'],['football', 'basketball', 'pingpang']]
    coordinates = {}
#    all_words = ['dog', 'cat', 'rabbit', 'football', 'basketball', 'pingpang']
    for topic in topics:
        for word in topic:
            x = random.random()
            y = random.random()
            coordinates[word] = (x,y)
#        draw_box.append({'x':axis_x, 'y':axis_y, 'word':all_words[i], 'color':random.choice(['green','blue'])})
    draw_topics(topics)
'''