
"""havest metadata from arxiv
source http://betatim.github.io/posts/analysing-the-arxiv/
Harvestes metadata from arxiv in order to find all papers in a category.
Attributes:
    ARXIV (str): link to the arxiv oai api
    OAI (str): link to oai
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

OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"

def harvest(arxiv="cs", startdate = "2012-01-01", enddate = "2019-08-31"): #physics:hep-ex
    """
    Harvestes metadata for a specific category on arxiv
    
    Args:
        arxiv (str, optional): category on arxiv (cs, physics:hep-ex)
    
    Returns:
        pandas dataframe: a dataframe with metadata harvested from arxiv
    """
    num_papers = 0
    papers = []
    base_url = "http://export.arxiv.org/oai2?verb=ListRecords&"
    url = (base_url +
           "from=%s&until=%s&"%(startdate,enddate) +
           "metadataPrefix=arXiv&set=%s"%arxiv)
    #url = 'https://fanyi.baidu.com/#en/zh/truncate'

    while True:
        print( "fetching", url)
        try:
            response = ur.urlopen(url)
            print('getting it')
        except:
            to = 10
            print("Got 503. Retrying after {0:d} seconds.".format(to))
            time.sleep(to)
            continue
            
        xml = response.read()

        root = ET.fromstring(xml)

        for record in root.find(OAI+'ListRecords').findall(OAI+"record"):
#            arxiv_id = record.find(OAI+'header').find(OAI+'identifier')
#            print(arxiv_id)
            meta = record.find(OAI+'metadata')
            
            info = meta.find(ARXIV+"arXiv")
            created = info.find(ARXIV+"created").text
            created = datetime.datetime.strptime(created, "%Y-%m-%d")
            categories = info.find(ARXIV+"categories").text
            #print(ET.tostring(info))
            authors = []
            for author in info.find(ARXIV+"authors").findall(ARXIV+"author"):
                a= {}

                a['keyname'] = author.find(ARXIV+"keyname").text

                try:
                    a['forenames'] = author.find(ARXIV+'forenames').text
                except AttributeError as e:
                    a['forenames'] = ''
                authors.append(a)
            # if there is more than one DOI use the first one
            # often the second one (if it exists at all) refers
            # to an eratum or similar
            doi = info.find(ARXIV+"doi")
            if doi is not None:
                doi = doi.text.split()[0]
            arxivid = info.find(ARXIV+"id").text
            arxivid = re.sub('/','',arxivid)
            contents = {'title': info.find(ARXIV+"title").text,
                        'arxivid': arxivid,
                        'abstract': info.find(ARXIV+"abstract").text.strip(),
                        'created': created,
                        'categories': categories.split(),
                        'doi': doi,
                        'authors' : authors
                        }

            papers.append(contents)
            num_papers += 1
#            with open(writer, 'a') as w:
#                s = '{}\t{}\t{}\n'.format(str(created), contents['abstract'], str(categories))
#                w.write(s)
#            exit()
				
        # The list of articles returned by the API comes in chunks of
        # 1000 articles. The presence of a resumptionToken tells us that
        # there is more to be fetched.
        token = root.find(OAI+'ListRecords').find(OAI+"resumptionToken")
        if token is None or token.text is None:
            break

        else:
            url = base_url + "resumptionToken=%s"%(token.text)
    print('there are {} papers in total.'.format(num_papers))        
    return papers


###############  Download 2012-2019 and pickle it #############################

#logtxt = 'mylog.txt'
'''
for year in range(2012, 2019):
    #writer = 'data/{}_papers.txt'.format(year)
    papers = harvest(arxiv="cs", startdate = "{}-01-01".format(year), enddate = "{}-12-31".format(year))
    with open('data/{}_papers.pkl'.format(year), 'wb') as w:
        pickle.dump(papers, w)
        w.close()'''
###############################################################################    
        
def rm_seg(string):
    text = string.strip().split()
    text = ' '.join(text)
    return text

def my_print(string, file):
    print(string)
    with open(file, 'a') as writer:
        time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        s = '{} {}\n'.format(time, string)
        writer.write(s)
    writer.close()

def pickle2txt(target_category='cs.LG'):
    flag = 'This article has been withdrawn' ; num_withdraw = 0
    num_abstract = 0
    filelist = []
    for year in range(2012,2020):
        file = os.path.join(os.getcwd(),'data/arxiv/{}_papers.pkl'.format(year))
        filelist.append(file)
    data_path = os.path.join(os.getcwd(), 'data/arxiv')
    w = os.path.join(data_path, 'train_{}.txt'.format(target_category))
    log = os.path.join(data_path, 'log.txt')
    for file in filelist:
        per_num_withdraw = 0
        per_num_abstract = 0
        papers = loadpk(file)
        with open(w, 'a') as writer:
            for paper in papers:
                time = paper['created']
                time = datetime.date(time.year, time.month, time.day)
                
                category = paper['categories']
                if target_category in category:
                    pass
                else:
                    per_num_withdraw += 1
                    continue

                text = paper['abstract']
                if text[:31] == flag:
                    per_num_withdraw += 1
                    continue
                else:
                    text = rm_seg(text)
                    s = '{}\t{}\t{}\n'.format(time, category, text)
                    writer.write(s)
                    per_num_abstract += 1
            my_print('harvest {} papers and throw {} papers from {}.'.format(
                    per_num_abstract, per_num_withdraw, file), log)
            num_withdraw += per_num_withdraw
            num_abstract += per_num_abstract
        writer.close()
    my_print('harvest {} papers and throw {} papers from all file.'.format(
                    num_abstract, num_withdraw), log)
        
    
        
if __name__ == '__main__':
    pickle2txt(target_category='cs.LG')
    pickle2txt(target_category='cs.CL')