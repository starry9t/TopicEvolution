#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:46:42 2019

@author: Yu Zhou
"""

import os
import sys

project_path = os.path.dirname(__file__)
sys.path.append(project_path)

import time
import datetime
import urllib.request as ur
import datetime
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

import numpy as np
import sys
import re
import pickle
from Utils.utils import cleanStr, loadpk, pk

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['ii','iii','etc','et','al','ie'])
stopwords = set(stop_words)
import enchant
endict = enchant.Dict("en_UK")
import subprocess
import arxiv as ax

suc_ids = []
faild_ids = []

def pickle2pdf(target_category='cs.LG'):
    #flag = 'This article has been withdrawn' 
    filelist = []
    for year in range(2012,2020):
        file = os.path.join(os.getcwd(),'data/arxiv/{}_papers.pkl'.format(year))
        filelist.append(file)
    print('filelist ready')
    num_suc = 0 ; num_fail = 0
    global c
    c = ''
    for f_i, file in enumerate(filelist):
        papers = loadpk(file)
        print('pickled paper loaded {}'.format(f_i))
        def custom_slugify(obj):
            name = obj.get('id').split('/')[-1]
            #time = obj.get('published').split('T')
            res = 'data/arxiv/{}/pdf/'.format(c) + name#+ time + '_' + name
            return  res#obj.get('id').split('/')[-1]
        
        for paper in papers:
            arxiv_id = paper['arxivid']
            category = paper['categories']
            if 'cs.LG' in category:
                if 'cs.CL' in category:
                    c = 'LG_CL'
                else:
                    c = 'LG'
            elif 'cs.CL' in category:
                c = 'CL'
            else:
                pass
            
            try:
                d_paper = ax.query(id_list=[arxiv_id])[0]
                ax.download(d_paper, slugify=custom_slugify)
#                res = 'data/arxiv/{}/'.format(target_category) + d_paper.get('id').split('/')[-1]
#                all_pdf_path.append(res)
                print('download {} {} succeed.'.format(arxiv_id, c))
                with open('suc_ids.txt', 'a') as w:
                    w.write('{}\t{}\n'.format(arxiv_id,c))
                    w.close()
                num_suc += 1
            except:
                print('----------download {} {} failed'.format(arxiv_id,c))
                with open('faild_ids.txt', 'a') as w:
                    w.write('{}\t{}\n'.format(arxiv_id,c))
                    w.close()
                num_fail += 1
    print('num_suc: {} , num_fail: {}'.format(num_suc, num_fail))
    return

if __name__ == '__main__':
    print('start')
    all_pdf_path = pickle2pdf(target_category='cs.LG')    