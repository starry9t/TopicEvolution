
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

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['ii','iii','etc','et','al','ie'])
stopwords = set(stop_words)
import enchant
endict = enchant.Dict("en_UK")
import subprocess
import arxiv as ax

OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"

def harvest(arxiv="cs", startdate = "2019-10-01", enddate = "2019-10-03"): #physics:hep-ex
    """
    Harvestes metadata for a specific category on arxiv
    
    Args:
        arxiv (str, optional): category on arxiv (cs, physics:hep-ex)
    
    Returns:
        pandas dataframe: a dataframe with metadata harvested from arxiv
    """
    num_papers = 0
    papers = []
    all_arxiv_ids = []
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
            arxivid = info.find(ARXIV+"id").text
            arxivid = re.sub('/','',arxivid)
            all_arxiv_ids.append(arxivid)
    return all_arxiv_ids

def pickle2pdf(target_category='cs.LG'):
    #flag = 'This article has been withdrawn' 
    num_withdraw = 0
    num_abstract = 0
    filelist = []
    all_pdf_path = []
    for year in range(2012,2013):
        file = os.path.join(os.getcwd(),'data/arxiv/{}_papers.pkl'.format(year))
        filelist.append(file)
    print('filelist ready')
    data_path = os.path.join(os.getcwd(), 'data/arxiv')
    w = os.path.join(data_path, 'train_{}.txt'.format(target_category))
    log = os.path.join(data_path, 'log.txt')
    n = 0
    for f_i, file in enumerate(filelist):
        per_num_withdraw = 0
        per_num_abstract = 0
        papers = loadpk(file)
        print('pickled paper loaded {}'.format(f_i))
        def custom_slugify(obj):
            name = obj.get('id').split('/')[-1]
            #time = obj.get('published').split('T')
            res = 'data/arxiv/{}/pdf/'.format(target_category) + name#+ time + '_' + name
            print(res)
            return  res#obj.get('id').split('/')[-1]
        
        for paper in papers:
            if n > 10:
                break
            arxiv_id = paper['arxivid']
            try:
                d_paper = ax.query(id_list=[arxiv_id])[0]
                ax.download(d_paper, slugify=custom_slugify)
#                res = 'data/arxiv/{}/'.format(target_category) + d_paper.get('id').split('/')[-1]
#                all_pdf_path.append(res)
            except:
                print('download {} failed'.format(arxiv_id))
            n+=1
    return

def convert_pdfs(pdf_folder):
    txt_folder = os.path.dirname(pdf_folder)
    txt_folder = os.path.join(txt_folder, 'txt')
    for root, dirs, files in os.walk(pdf_folder):
        for file in files:
            pdf_filename = os.path.join(root, file)
            txt_filename = os.path.join(txt_folder, file)
            print(pdf_filename)
            print(txt_filename)
            exit()
    return

def pdf2txt(pdf_file, txt_file, deli=''):

    cmd = "pdf2txt.py -V tt.pdf" #-t text -o text.txt -c utf-8 
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    res = subp.stdout.readlines()
    
    n = 0
    with open('text.txt', 'w', encoding='utf-8') as w:
        for b_ in res:
#            if n>50:
#                w.close()
#                exit()
    #        if type(b_) == bytes:
            s = b_.decode('utf-8')
            s = keep_char(s.strip())
    #            print(s)
    #            print(type(s))
            s = raw_clean(s, deli=deli)
#            print(s)
            for token in s.split(' '):
                w.write(token)
                w.write(' ')
#            if s == ' ':
#                continue
#            w.write(s.strip())
#            w.write('\n')
            n+=1
                #exit()
        w.close()
    return

        
def url2pdf(arxiv_id):
#    pdf_link = 
#    ax.download(paper)
    return

def keep_char(string):
    new_string = ''
    for char in string:
        if char.isalpha():
            new_string += char
            continue
        if char == ' ':# in ['λ','θ','^']:
            new_string += char
            continue
    return new_string

def raw_split(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)

def raw_clean(s, deli):
    #tokens = s.split(' ')
    tokens = raw_split(delimiters=deli, string=s, maxsplit=0)
    newstring = []
    for token in tokens:
        if not token:
            continue
        if len(token) == 1:
            continue
        if token[-1] == 'i':
            continue
        if token[-1] == 'j':
            continue
        if token in stopwords:
            continue
#        if token.islower():
#            if endict.check(token):
#                newstring.append(token)
#        else:
#            newstring.append(token)
        newstring.append(token)
    res = ' '.join(newstring)  
    return res #newstring # res


        
        
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


        
    
        
if __name__ == '__main__':
    print('start')
    all_pdf_path = pickle2pdf(target_category='cs.LG')    
#    convert_pdfs(all_pdf_path)
    
    
    ########################################
    
    # download contents
    #test
#    papers = harvest(arxiv="cs", startdate = "2019-10-01", enddate = "2019-10-03")
#    print(len(papers))
#    
#    download_papers(papers)
#    for year in range(2012, 2019):
#        #writer = 'data/{}_papers.txt'.format(year)
#        papers = harvest(arxiv="cs", startdate = "{}-01-01".format(year), enddate = "{}-12-31".format(year))
#        with open('data/{}_papers.pkl'.format(year), 'wb') as w:
#            pickle.dump(papers, w)
#            w.close()
        
#    pickle2txt(target_category='cs.LG')
#    pickle2txt(target_category='cs.CL')

    
    # download pdf
    
    
    #import urllib
#    url = 'http://export.arxiv.org/oai2?verb=ListRecords&set=math&from=2015-01-01&until=2015-01-02&metadataPrefix=arXiv'
#
#    data = ur.urlopen(url).read()
#    print(data)
    exit()
    delimiters = ' ', '\t', '\n', '-'
    pdf2txt('','',deli=delimiters)
    
    '''
    def harvest(arxiv="cs", startdate = "2019-10-01", enddate = "2019-10-31"): #physics:hep-ex
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
            print('info:')
            print(info)
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
            print('arxivid: ', arxivid)
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
            break
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
    return papers'''
    
'''
def pickle2txt(target_category='cs.LG'):
    #flag = 'This article has been withdrawn' 
    num_withdraw = 0
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
                front_line = text.strip().split(' ')[:12]
                if 'withdrawn' in front_line:
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
'''