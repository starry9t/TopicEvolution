#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:16:21 2019

@author: Yu Zhou
"""
import os
import sys
# remove this part after testing
project_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_path)

#import time
import datetime

from Utils.utils import cleanStr, pk, loadpk, trace, key_words
from Utils.utils import get_cohs, display_topics

import math
from collections import Counter, defaultdict

from gensim import corpora
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
from gensim.matutils import hellinger

def to_datetime(string, dataname='arxiv'):
    try:
        if dataname == 'arxiv':
            ymd = string#.strip().split('-')            
        elif dataname == 'care':
            ymd = string.strip().split('T')[0]
        year, month, day = ymd.strip().split('-')
        dt = datetime.date(int(year),int(month),int(day))
        return dt
    except:
        return None
    
def trans_datetime(cur_date=datetime.date.today(), unit='month', 
                   time_period=1, start_date=datetime.date(1999,9,9)):
    if unit == 'year':
        new_date = cur_date.year
        return new_date
    elif unit == 'month':
        month_remain = 0
        month_remain = (cur_date.month+11) % time_period
        new_month = cur_date.month - month_remain
        new_date = cur_date.year*100 + new_month
        return new_date
    elif unit == 'day':
        time_duration = (cur_date - start_date).days
        remainder = time_duration % time_period
        new_date = cur_date-datetime.timedelta(days=remainder)
        return new_date
    else:
        print('Please enter the right type of time unit. \
              It should be \'year\', \'month\', or \'day\'.')
        return None#datetime.date.today()
    
def get_vm(config):
    if config.vocab_method == 'topk':
        return config.vocab_method, config.reserve_vocab
    elif config.vocab_method == 'all':
        return config.vocab_method, 0
    elif config.vocab_method == 'highlow':
        return config.vocab_method , (config.highlow_high, config.highlow_low)
    elif config.vocab_method == 'tfidf':
        return config.vocab_method , config.reserve_vocab
    else:
        raise TypeError('config.vm does not exist!')

class Name(object):
    def __init__(self, flag="", config="", model_name="", data_name="", time_slices=[]):
        self.flag = flag
        self.config = config
        self.model = model_name
        self.data = data_name
        self.time_slices = time_slices
        #self.tw

    def show(self):
        return '[flag:{} , config:{} , model_name:{} , data_name:{}]'.format(
                self.flag, self.config, self.model, self.data)

class Result(object):
    def __init__(self, info=Name(), tag='init', TWmatrix=[], itow={},
                 twlist=[], COHs=[], PPLs=[] ):
        self.info = info
        self.tag = tag
        self.TWmatrix = TWmatrix # 
        self.itow = itow # or [itow0, itow1, ...]
        self.twlist = twlist
        self.COHs = COHs
        self.PPLs = PPLs
        
    def add_cohs(self, file=None):
        if len(self.COHs)>1:
            print('result.COHs already exists!')
        else:
            for n,tw_list in enumerate(self.twlist):
                cohs = get_cohs(tw_list)
                self.COHs.append(cohs)
                print('cohs {}/{}'.format(n+1,len(self.twlist)))
        if file is None:
            return
        else:
            t = 0
            for tw_list, cohs in zip(self.twlist, self.COHs):
                seg = '------ topics in time {}  ------'.format(t)
                self.write_t(file, tw_list, cohs, seg)
                t += 1                
    def write_t(self, writer, tw_list, cohs=None, seg='---', head='topic'):
        display_topics(tw_list=tw_list, cohs=cohs, head=head, file=writer)
        
    def get_evos(self, file=None):
        num_topics = len(self.twlist[0])
        topic_evos = [] ;  cohs_evos = []
        for i in range(num_topics):
            topic_evos.append([])
            cohs_evos.append([])
        for twlist, cohs in zip(self.twlist, self.COHs):
            for topic_idx, words in enumerate(twlist):
                topic_evos[topic_idx].append(words)
                c = cohs[topic_idx]
                if type(c)==str:
                    c = c[:-5]
                cohs_evos[topic_idx].append(float(c))
        if file is not None:
            t = 0
            for topic_evo, cohs_evo in zip(topic_evos, cohs_evos):
                seg = '------ topic {} evolution ------'.format(t)
                self.write_t(file, topic_evos, cohs_evos, seg)
                t += 1
        return topic_evos, cohs_evos
        
    def T(self):
        return len(self.info.time_slices) 
    
    def get_avg_cohs(self):
        avg_cohs = []
        for cohs in self.COHs:
            avg_coh = sum(cohs)/len(cohs)
            avg_cohs.append(avg_coh)
        return avg_cohs
    
    def show_avg_cohs(self):
        s = ['%.6f'%i for i in self.get_avg_cohs()]
        return ' '.join(s)
    
    def show_best_topic(self, time=None):
        
        if time is not None:
            print('get best topic in time {}'.format(time))
            the_cohs = [cohs[time] for cohs in self.COHs]
            new_cohs = []
            for item in the_cohs:
                if type(item) == str:
                    item = item[:-5]
                new_cohs.append(float(item))            
            max_coh = max(new_cohs)
            topic_idx = new_cohs.index(max(new_cohs))
            tw = self.twlist[topic_idx][time]
            print('done ... get best topic in time {}'.format(time))
            return tw, max_coh
        
        # else
        max_indices = [] ; max_cohs=[]
        for cohs in self.COHs:
            new_cohs = []
            for item in cohs:
                if type(item) == str:
                    item = item[:-5]
                new_cohs.append(float(item))
            max_coh_idx = new_cohs.index(max(new_cohs))
            max_indices.append(max_coh_idx)
            max_cohs.append(new_cohs[max_coh_idx])   
        topic_idx = max_cohs.index(max(max_cohs))
        max_coh = max(max_cohs)
        tw = self.twlist[topic_idx][max_indices[topic_idx]]
        return tw, max_coh
    
    def get_topic_of(self, topic_name='algorithm', time=None):
        if time is not None:
            return None
        for twlist in self.twlist:
            for topic in twlist:
                for word in topic:
                    print(word, topic_name)
                return
        return

class Dataset(object):
    def __init__(self, config=None,
                 start_date = datetime.date(2099, 9, 9),
                 end_date = datetime.date(1900, 1, 1),
                 set_start = datetime.date(2012, 1, 1),
                 set_end = datetime.date(2099, 9, 9)
                 ):#, unit='day', period=1):
        self.config = config
        self.start_date = start_date
        self.end_date = end_date
        self.set_start = set_start
        self.set_end = set_end
        self.times = []
        self.docs = []
        self.counter = Counter()
        self.cur_counter = None
#        self.unit = unit
#        self.period = period
        self.itow = {}
        self.wtoi = {}
        
        if self.config is not None:
            self.all_set()
            
    def all_set(self):
        self.read(file=self.config.train_file, dataname=self.config.dataname)
        #self.idf_words()
        method, num = get_vm(self.config)
        self.update_counter(method, num)
        self.make_vocab()
        self.update_dataset()    
        self.seg() 
        
        
    def read(self, file, dataname='arxiv', prepcs=False):
        if dataname == 'arxiv':
            with open(file, 'r') as reader:
                if prepcs:
                    pass
                else:
                    line_counter = 0 ; valid_counter = 0
                    for line in reader:
                        line_counter += 1
                        tokens = line.strip().split('\t')
                        time = tokens[0]
                        text = tokens[-1]
                        time = to_datetime(time, dataname = 'arxiv')
                        if time == None:
                            continue
                        if time<self.set_start or time>self.set_end:
                            continue
                        text, wc = cleanStr(text, self.config.deli)
                        if wc < self.config.min_num_words:
                            continue
                        self.times.append(time)
                        self.docs.append(text)  
                        self.counter.update(text.strip().split())
                        valid_counter += 1
                        if time < self.start_date:
                            self.start_date = time
                        elif time > self.end_date:
                            self.end_date = time                    
            reader.close()
            info = 'extract {}/{} lines from {}'.format(valid_counter, line_counter, file)
            trace(info, self.config.log_file)
            
        elif dataname == 'care':
            with open(file, 'r') as reader:
                if prepcs:
                    pass
                else:
                    line_counter = 0 ; valid_counter = 0
                    for line in reader:
                        line_counter += 1
                        tokens = line.strip().split('\t')
                        time = tokens[0]
                        if len(tokens) == 2:
                            text = tokens[-1]
                        elif len(tokens) == 3:
                            text = tokens[1] + ' ' + tokens[2]
                        else:
                            continue
                        time = to_datetime(time, dataname='care')
                        if time == None:
                            continue
                        if time<self.set_start or time>self.set_end:
                            continue
                        text, wc = cleanStr(text)
                        if wc < self.config.min_num_words:
                            continue
                        self.times.append(time)
                        self.docs.append(text)  
                        self.counter.update(text.strip().split())
                        valid_counter += 1
                        if time < self.start_date:
                            self.start_date = time
                        elif time > self.end_date:
                            self.end_date = time    
            reader.close()
            info = 'extract {}/{} lines from {}'.format(valid_counter, line_counter, file)
            trace(info, self.config.log_file)           
            
        else:
            info = 'unrecgonized argument dataname'
            trace(info, self.config.log_file)
            pass
        

        
    def update_counter(self, method='topk', num=1):
        # method belongs in ['topk', 'highlow', 'all']
        if method == 'topk':
            assert type(num) == int , 'num should be an int!'
            self.cur_counter = Counter()
            for (k,v) in self.counter.most_common(num):
                self.cur_counter[k] = v
        elif method == 'all':
            self.cur_counter = self.counter
        elif method == 'highlow':
            # num shoule be tuple like (0.1, 0.3)
            assert type(num) == tuple , 'num should be a tuple!'
            assert sum(num) < 1 , 'elements of num should be a percent value and the sum should less than 1!'
            total_num = len(self.counter)
            front_num = math.ceil(total_num*num[0])
            rear_num = math.ceil(total_num*num[1])
            temp_c = Counter(self.counter.most_common(total_num-rear_num))
            front_c = Counter(self.counter.most_common(front_num))
            self.cur_counter = temp_c - front_c
            self.cur_counter += Counter() # remove 0 and negatives
#        elif method == 'tfidf':
#            self.cur_counter = Counter()
        elif method == 'tfidf':
            self.cur_counter = key_words(self.docs, n_cross=num)                     
        else:
            raise TypeError('method should be one of [\'topk\', \'all\',\'highlow\']')
        self.vocab = set(self.cur_counter.keys())
        
#        file = 'k_words_{}.txt'.format(self.config.filename)
#        with open(file, 'w') as writer:
#            for word in k_words:
#                writer.write('{}\n'.format(word))
#            writer.close() 
        
    def make_vocab(self):
        if self.cur_counter is None:
            pass
        else:    
            i = 0
            for w in self.cur_counter.keys():
                self.itow[i] = w
                self.wtoi[w] = i
                i += 1
            trace('made vocab size {}'.format(len(self.itow)), self.config.log_file)

    def update_dataset(self):
        new_times = [] ; new_docs = []
        old_c = len(self.times)
        for time, doc in zip(self.times, self.docs):
            new_doc = [word for word in doc.split() if word in self.vocab]
            if len(new_doc)<self.config.min_num_words:
                continue
            else:
                new_times.append(time)
                new_docs.append(new_doc)
        self.times = new_times
        self.docs = new_docs
        new_c = len(self.times)
        trace('update dataset from {} to {}'.format(old_c, new_c), self.config.log_file)  

      
    def seg(self, unit=None, time_period=None):
        
        # according to 'self' unit and period
        d = {}
        for t_i, time in enumerate(self.times):
            new_datetime = trans_datetime(cur_date=time, unit=self.config.unit,
                                          time_period=self.config.time_period,start_date=self.start_date)
            if new_datetime in d:
                d[new_datetime].append(t_i)
            else:
                d[new_datetime] = [t_i]
        #d = list(filter(lambda x:len(x[1])>self.config.min_num_docs, d.items()))
        d = sorted(d.items(), key=lambda x:x[0])
        #print(d)
        self.sorted_idx = []
        self.time_slices = []
        self.times_tag = []
        for i in range(len(d)-1):
            if len(d[i][1])<self.config.min_num_docs:
                continue
            self.time_slices.append(len(d[i][1]))
            self.sorted_idx += d[i][1]
            self.times_tag.append(str(d[i][0]))
        # cut for test set
        i = -1 ;  cut=2000
        if len(d[i][1])<cut:
            self.time_slices.append(len(d[i][1]))
            self.sorted_idx += d[i][1]
            self.times_tag.append(str(d[i][0])+'_test')
        else:
            self.time_slices.append(len(d[i][1][:-cut]))
            self.sorted_idx += d[i][1][:-cut]
            self.times_tag.append(str(d[i][0]))
            
            self.time_slices.append(len(d[i][1][-cut:]))
            self.sorted_idx += d[i][1][-cut:]
            self.times_tag.append(str(d[i][0])+'_test')          
            
        trace('seg documents({}) to {} slices [{}] based on [{}]'.format(
                sum(self.time_slices), len(self.time_slices),
                ','.join([str(i) for i in self.time_slices]),
                ','.join(self.times_tag)),
                self.config.log_file)
    
    def vocab_size(self):
        return len(self.wtoi)
        
    def T(self):
        return len(self.time_slices)-1
    
    def save(self):
        pass
    
    def load(self, alist):
        pass
    
    def __iter__(self):
        start_idx = 0
        for time_slice in self.time_slices[:-1]:
            cur_idx = self.sorted_idx[start_idx : start_idx+time_slice]
            start_idx += time_slice
            yield [self.docs[idx] for idx in cur_idx]
    # self.docs[idx] for idx in self.sorted_idx
    def test(self):
        n = self.time_slices[-1]
        return [self.docs[idx] for idx in self.sorted_idx[-n:]]
    
    def tfidf(self, docs=[['natural','language','processing']]):
        # return a dict D={'word': itidf_value , ...}
        # abstracts do not have much differences in term frequency.
        
        start_idx = 0
        for time_slice in self.time_slices[:-1]:
            cur_idx = self.sorted_idx[start_idx : start_idx+time_slice]
            start_idx += time_slice
            cur_docs = [self.docs[idx] for idx in cur_idx]
        
        return
    
    def get_doc_vectors(self, docs=None):
        if docs == None:
            docs = self.docs
        for doc in docs:
            pass
        return
            
#    def get_term_freq(self, docs=None, words=None):
#        if docs==None:
#            docs=self.docs
#        if words==None:
#            words=set(self.counter.keys())
#        
    
#    def get_file_freq(self, docs=None, words=None):
#        if docs==None:
#            docs=self.docs
#        if words==None:
#            words==set(self.counter.keys())
#        file_freq = Counter()
#        for doc in docs:
#            if type(doc) == str:
#                doc_words = set(doc.split())
#            elif type(doc) == list:
#                doc_words = set(doc)
#            else:
#                raise TypeError("doc type should be str or list instead of {}".format(type(doc)))
#            for word in doc_words:
#                if word in file_freq:
#                    file_freq[word] += 1
#                else:
#                    file_freq[word] = 1
#        return file_freq
            
        
    def docs_sims(self, word_emd ,docs=None):
        # return a matrix of DxD
        if docs==None:
            docs = self.docs
        
    
    
def numerate_dataset(dataset_iter, wtoi, vocab_size=1):
    res = []
    for doc in dataset_iter:
        res_i = [0.0]*vocab_size
        for word in doc:#.strip().split(' '):
            try:
                res_i[wtoi[word]] += 1
            except:
                pass
        res.append(res_i)
    return res
    
class GensimData(object):
    def __init__(self, docs=None, test=[], wtoi={}):
        if docs == None:
            self.form_test(test, wtoi)
            return
        
        self.texts = docs
        self.test = test
        self.make()
#        self.texts = []
#        self.corpus = []
#        #self.itow = {}
#        #self.wtoi = {}
#        if docs is not None:
#            self.read_docs(docs)
#            self.make()      
        
    def read_file(self, file, prepcs=False):
        with open(file, 'r') as reader:
            if prepcs:
                # clean strings / remove stopwords , etc..
                pass
            else:
                for line in reader:
                    self.texts.append(line.strip().split())
            reader.close()
            
    def read_docs(self, docs):
        if len(self.texts) > 1:
            info = input('Data.texts is not empty, sure to continue? [y/n]')
            if info == 'y':
                pass
            else:
                return
            
#        for doc in docs:
#            self.texts.append(doc.strip().split())
        self.texts = docs
        
            
    def make(self):
        self.dictionary = corpora.Dictionary(self.texts+self.test)       
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        self.test = [self.dictionary.doc2bow(text) for text in self.test]
        
    def __iter__(self):
        for i in range(len(self.corpus)):
            yield self.corpus[i]
        
def form_gensim_docs(docs, wtoi):
    new_docs = []
    for doc in docs:
        new_doc = []
        # doc = [word1, word2, ... ]
        c = Counter(doc)
        for word, count in c.items():
            word_idx = wtoi[word]
            new_doc.append((word_idx, count))
        new_docs.append(new_doc)
    return new_docs
    pass