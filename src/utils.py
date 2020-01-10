#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:45:16 2019

@author: Yu Zhou
"""

import os
import re
import time
#import sys
#import nltk
import datetime
import math
import numpy as np
import torch
import pickle
#import random
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['ii','iii'])
stopwords = set(stop_words)
import enchant
endict = enchant.Dict("en_UK")
#from palmettopy.palmetto import Palmetto
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import subprocess
from collections import Counter

log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),'log.txt')

project_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
git_path = os.path.abspath(os.path.dirname(project_path))
pmt_path = os.path.abspath(os.path.join(git_path, 'palmetto_java'))
pmt_java = os.path.abspath(os.path.join(pmt_path, 'palmetto-0.1.0-jar-with-dependencies.jar'))
wiki_bd = os.path.abspath(os.path.join(pmt_path, 'wikipedia_bd'))
topic_file = os.path.join(pmt_path, 'topic')

def get_line_ge(line_path, network_file, embedding_file, emb_dim):
    cmd = '{} -train {} -output {} -size {} -order 2 -negative 5 -samples 5 -rho 0.025 -threads 20'.format(
            line_path, network_file, embedding_file, emb_dim)
    print('cmd: {}'.format(cmd))
    #os.popen(cmd)
    #subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    subprocess.call(cmd, shell=True)
    #res = subp.stdout.readlines()  
#    if subp.stdin:
#        subp.stdin.close()
#    if subp.stdout:
#        subp.stdout.close()
#    if subp.stderr:
#        subp.stderr.close()
#    try:
#        subp.kill()
#    except OSError:
#        pass
    return #res

def draw_tab(list_of_list, last_col=None, last_col_name='N/A',
             head='head', row_names=None, col_names=None, seg='seg',
             file=None):
    from prettytable import PrettyTable
    num_rows = len(list_of_list) ; num_cols = len(list_of_list[0])
    if last_col is not None:
        num_cols += 1  
        
    if col_names is None:
        col_names = ['col_{}'.format(i+1) for i in range(num_cols)]
    else:
        col_names += [last_col_name]
        
    if row_names is None:
        row_names = ['col_{}'.format(i+1) for i in range(num_rows)]
        
    header = [head]
    header += col_names
    tab = PrettyTable(header)
    tab.align[head] = 'r'
    for label in header[1:]:
        tab.align[label] = 'l' 
        
    for row_idx in range(num_rows):
        row = [row_names[row_idx]]
        row_list = list_of_list[row_idx]
        row += [str(item) for item in row_list]
        if last_col is not None:
            row.append(last_col[row_idx])
        tab.add_row(row)        
        
    if file is not None:
        with open(file, 'a') as w:
            w.write('---------- {} ----------\n'.format(seg))
            w.write(str(tab))
            w.write('\n\n')
            w.close()
            
    return tab

def write_topic(tw):
    with open(topic_file, 'w') as w:
        for words in tw:
            w.write('{}\n'.format(' '.join(words)))
        w.close()

def get_cohs(tw):
    write_topic(tw)
    cohs = []
    
    cmd = 'java -jar {} {} C_V {}'.format(pmt_java, wiki_bd, topic_file)
    print(cmd)
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = subp.stdout.readlines()
    
    N = len(res)-1
    n = 1
    for item in res[1:]:
        # items is i_th topic coh formatted as No. \t coh \t [word1, word2, ..., wordN] \n
        s = item.decode()
        tokens = s.strip().split('\t')
        coh = float(tokens[1])
        cohs.append(coh)
        #print('coh {}/{}'.format(n,N))
        n+=1
    #subp.kill()
    return cohs

def get_avg_cohs_of_tws(tws):
    avg_cohs = []
    for tw in tws:
        cohs = get_cohs(tw)
        avg_coh = sum(cohs)/len(cohs)
        avg_cohs.append(avg_coh)
    return avg_cohs  

###############################################################################
def ppl(DOCs, TW_tensor):
    # DOCs = [doc1, doc2, ... , docD]
    word_count = 0
    prob = 0.0
    for doc in DOCs:
        # doc = [(word_idx_1, word_idx_1_count), ... , (word_idx_Nd, word_Nd_count)]
        prob_doc = 0.0
        for word_id, word_id_count in doc:
            word_count += word_id_count
            prob_word = TW_tensor[:,word_id].sum()
            prob_doc += prob_word*word_id_count
        prob += prob_doc
    perplexity = math.exp(-prob/word_count)
    return perplexity
      

def draw_ppl(ppls, label='label', title='plot', file=None):
    times = len(ppls)
    x_labels = [str(i) for i in range(times)]    
    plt.plot(x_labels, ppls, color='blue', label=label)
    
    plt.legend()  
    plt.xticks(x_labels, x_labels, rotation=1)
     
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('times') 
    plt.ylabel("ppl") 
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    plt.title(title) 
    plt.savefig(file,dpi = 900)
    return
###############################################################################
    
def get_tw_list(tw_tensor, itow, k=10):
    tw_list = []
    (prob, wordIdx) = torch.topk(tw_tensor, k)
    for i in range(tw_tensor.size(0)):
        words = [itow[int(idx)] for idx in wordIdx[i]]
        tw_list.append(words)
    return tw_list

def batch_to_tensor(batch, useGPU=False):
    #tensor = torch.zeros((len(batch),len(batch[0])))
    texts = np.array(batch, dtype=np.float32)
    tensor = torch.from_numpy(texts).float()

    if useGPU:
        tensor = tensor.cuda()
    return tensor

def updateVocabs(vocabs = [], include_special=True):
    final_wtoi = {}
    for vocab in vocabs:
        final_wtoi.update(vocab.wtoi)
    vocab_size = 0
    if include_special:
        specials = ['<pad>', '<unk>', '<s>', '</s>']
        for special in specials:
            final_wtoi[special] = vocab_size
            vocab_size += 1     
    for word in final_wtoi.keys():
        if word not in specials:
            final_wtoi[word] = vocab_size
            vocab_size += 1
        
    final_itow = {i:w for w,i in final_wtoi.items()}
    return final_wtoi, final_itow, vocab_size
  

#######################   OUT PUT #############################################
    
def to_evo(tws):
    times = len(tws)
    num_topics , num_words= tws[0].size()
    new_tws = {}
    for topic_idx in range(num_topics):
        tw = torch.zeros((times, num_words))
        for time in range(times):
            tw[time] = tws[time][topic_idx]
        new_tws[topic_idx] = tw
    return new_tws



def display_topics(tw_list, cohs=None, coh_name='coh',
                   head='head', seg='---seg---', 
                   col_name='word', row_name='topic', file=None):
    from prettytable import PrettyTable
    num_rows = len(tw_list)
    num_cols = max([len(i) for i in tw_list])
    header = [head]
    for col in range(num_cols):
        header.append('{}_{}'.format(col_name, col+1))
    if cohs == None:
        cohs = ["-"]*num_rows
        header.append(coh_name)
    else:
        avg_coh = '%.6f' % (sum(cohs)/len(cohs))
        max_idx = cohs.index(max(cohs))
        cohs[max_idx] = '{}(max)'.format(cohs[max_idx])
        header.append('{}(avg={})'.format(coh_name,avg_coh))

    tab = PrettyTable(header)
    tab.align[head] = 'r'
    for label in header[1:]:
        tab.align[label] = 'l'    
    for row_idx in range(num_rows):
        row = ['{}_{}'.format(row_name,row_idx+1)]
        row += tw_list[row_idx]
        if len(row) < num_cols+1:
            row += [""]*(num_cols+1-len(row))
        row.append(cohs[row_idx])
        tab.add_row(row)

    if not file is None:
        with open(file, 'a') as w:
            w.write('---------- {} ----------\n'.format(seg))
            w.write(str(tab))
            w.write('\n\n')
        w.close()
    return tab
    
   
def tw2topics(twtensor, k=1):
    topics = []
    (prob, wordIdx) = torch.topk(twtensor, k)
    for idx in wordIdx:
        topics.append([int(i) for i in idx])
    return topics

def idx2words(topics, itow):
    new_topics = []
    for topic in topics:
        new_topics.append([ itow[i] for i in topic ])
    return new_topics

def topic2file(topics, file, info):
    with open(file, 'a') as writer:
#        writer.write('-----  {}  -----'.format(time.asctime(
#                        time.localtime(time.time()))))
        writer.write('{}\n'.format(info))
        for topic in topics:
            for word in topic:
                writer.write('{}\t'.format(str(word)))
            writer.write('\n')
    writer.close()    

def tw2file(twtensor, k=1, itow={}, file='temp.txt', info='--'):              #
    topics = tw2topics(twtensor, k=k)                                         #
    topics = idx2words(topics, itow)  
                                            #
    topic2file(topics, file, info)  
    #
                                          #

def topic_file_transpose(reader, writer, num_topic):
    topic_evo = []
    for i in range(num_topic):
        topic_evo.append([])
    n = -1
    with open(reader, 'r') as r:
        for line in r:
            if line[:5] == '-----':
                n += 1
                topic_idx = 0
            else:
                #words = line.strip().split()
                topic_evo[topic_idx].append(line.strip())
                topic_idx+=1
        r.close()
    with open(writer, 'w') as w:
        for i in range(num_topic):
            w.write('--------- Topic {} Evolution ----------\n'.format(i+1))
            for t, topic in enumerate(topic_evo[i]):
                w.write('time {}: {}\n'.format(t, topic))
        
##################################################################
                
def avg_tensor(tensor):
    new_tensor = torch.zeros((1,tensor.size(1)))
    for c in range(tensor.size(1)):
        new_tensor[0][c] = torch.sum(tensor[:,c])/tensor.size(0)
    return new_tensor


######################  preprocess  #########################



def mysplit(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemSent(tokens):
    #tokens = word_tokenize(string)
    tagged_sent = pos_tag(tokens)
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) 
    s = " ".join(lemmas_sent)
    return s    
    
def cleanStr(s, deli='uni'):
    if deli=='uni':
        delimiters = ' ','\t','\n',',','.','?','!','\'','’','/','(',')',"\"",'\\','{','}','[',']',':',';','0','1','2','3','4','5','6','7','8','9','%','$','#','*','-'
    elif deli == 'phrase':
        delimiters = ' ','\t','\n',',','.','?','!','\'','’','/','(',')',"\"",'\\','{','}','[',']',':',';','0','1','2','3','4','5','6','7','8','9','%','$','#','*','--','---'        
    elif deli == 'default':
        delimiters = ' ', '\t', '\n', '-'
    strs = mysplit(delimiters=delimiters, string=s, maxsplit=0)
    lowstrs = []
    for astr in strs:
        if astr:
            lowstrs.append(astr.lower())
        else:
            pass
    s = lemSent(lowstrs)
    newstring = []
    for astr in s.split(' '):
        try:
            if len(astr) == 1:
                continue
            elif astr not in stopwords:
                newstring.append(astr)
#                if endict.check(astr):
#                    newstring.append(astr)
        except:
            pass
    ns = ' '.join(newstring)
    return ns, len(ns)

###############################################################################
    
def get_file_freq(docs=None, words=None):
    file_freq = Counter()
    for doc in docs:
        if type(doc) == str:
            doc_words = set(doc.split())
        elif type(doc) == list:
            doc_words = set(doc)
        else:
            raise TypeError("doc type should be str or list instead of {}".format(type(doc)))
        for word in doc_words:
            if word in file_freq:
                file_freq[word] += 1
            else:
                file_freq[word] = 1
    return file_freq

def ff2idf(ff, D=1):
    idf_dict = {}
    for key in ff.keys():
        idf_dict[key] = math.log(D/ff[key])
    return idf_dict
        
def tfidf_dict(term_freq, idf_freq):
    res = {}
    for key in term_freq.keys():
        res[key] = term_freq[key] * idf_freq[key]
    return res

def most_common_keys(counter, n):
    res = list()
    tuples = counter.most_common(n)
    for key, value in tuples:
        res.append(key)
    return res

def key_words(docs, words=None, n_cross=3000):
    if words == None:
        all_words = Counter()
        for doc in docs:
            doc = doc.split(' ')
            all_words.update(doc)
        words = set(all_words.keys())
    file_freq = get_file_freq(docs, words)
    idf_dict = ff2idf(file_freq, D=len(docs))
    tfidf = tfidf_dict(all_words, idf_dict)
    
#    words_1 = most_common_keys(all_words, n_cross)
#    words_2 = most_common_keys(file_freq, n_cross)
#    cross_words = words_1 & words_2
    tfidf_d = sorted(tfidf.items(), key=lambda d: d[1], reverse=True) 
    res_c=Counter()
    for i, (k,v) in enumerate(tfidf_d):
        if i>n_cross:
            break
        res_c[k] = v
#    files = ['1.txt','2.txt','3.txt']
#    wordss = [words_1, words_2, cross_words]
#    for file, words in zip(files, wordss):
#        with open(file, 'w') as writer:
#            for w in words:
#                writer.write('{}\n'.format(w))
#            writer.close()        
    return res_c
###############################################################################
    
def pk(file, data):
    with open(file, 'wb') as w:
        pickle.dump(data, w)
        w.close()
    return

def loadpk(file):
    with open(file, 'rb') as r:
        data = pickle.load(r)
        r.close()
    return data

def trace(string, file=None, write='a'):
    print(string)
    if file is None:
        return
    with open(file, write) as r:
        time = datetime.datetime.now()
        s = '{}\t{}\n'.format(str(time), string)
        r.write(s)
        r.close()
    
def check_path(path, action='add'):
    if not os.path.exists(path):
        if action == 'add':
            os.makedirs(path)
        elif action == 'report':
            trace('{} does not exist! please check!'.format(path))
            exit()
            
###############################################################################
            
def make_glove_embeddings(e_file, wtoi, x_dim, h_dim):
    count = 0
    weight = torch.zeros(x_dim, h_dim)
    with open(e_file, 'r') as reader:
        for line in reader:
            tokens = line.strip().split(' ')
            if len(tokens) != 101:
                pass
            else:
                word = tokens[0]
                embedding = [float(item) for item in tokens[1:]]
                if word in wtoi:
                    idx = wtoi[word]
                    weight[idx] = torch.tensor(embedding)
                    count += 1
        reader.close()
    s = '{}/{} words\' * {} embedding has been initialized.'.format(count, x_dim, h_dim)
    print(s)
    return weight