#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:17:33 2019

@author: Yu Zhou
"""

import os
import time
import urllib
#from lxml import html
from bs4 import BeautifulSoup
from progressbar import Bar, ETA, FileTransferSpeed, Percentage, ProgressBar

cur_dir = os.path.dirname(__file__)
urlFile = os.path.join(cur_dir, 'url.txt')
opinionFile = os.path.join(cur_dir, 'careopinion.txt')

logFile = os.path.join(cur_dir, 'log.txt')



def getTime():
    return(time.strftime('%m-%d %H:%M:%S',time.localtime(time.time())))
    
def log(text):
    text = '{} {}\n'.format(getTime(),text)
    with open(logFile, 'a') as writer:
        writer.write(text)
    writer.close()

def getUrls(urltxt):
    urls = []
    with open(urltxt, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            url = line.strip()
            urls.append(url)
    reader.close()
    return urls

def url2html(url):
    try:
        response = urllib.request.urlopen(url, timeout=3)
        htmlDoc = response.read()
        return htmlDoc
    except:
        return False
        
def html2txt(htmlDoc):
    soup = BeautifulSoup(htmlDoc, 'lxml')
    try:
        atime = soup.find("time")['datetime']
        review = soup.find(id="opinion_body").get_text()
        try:
            reply = soup.find("div", {"class" : " tm-area"}).get_text()
        except:
            reply = ''
        line = '{}\t{}\t{}\n'.format(atime.strip(), review.strip(), reply.strip())
        return line
    except:
        return False
        
def urls2file(urls, file):
    lostUrls = []
    with open(file, 'w') as writer:    
        widgets = ['{processing}: ', Percentage(), ' ', Bar(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=(len(urls)))
        pbar.start()
        num = 0
        for url in urls:
            htmlDoc = url2html(url)
            if htmlDoc:
                txt = html2txt(htmlDoc)
                if txt:
                    writer.write(txt)
            else:
                lostUrls.append(url)
            num += 1
            pbar.update(num)
        pbar.finish()
    writer.close()
    return lostUrls

def main(urls, file):
    count = 1
    while len(urls)>0:
        s = '{} : start scrapping urls of length {}'.format(count, len(urls))  ; count += 1
        print(s) ; log(s)
        urls = urls2file(urls, file)
    s = 'all urls have been scrapped into careopinion.txt'
    print(s) ; log(s)
    
if __name__ == "__main__":
    urls = getUrls(urlFile)
    print('there are {} urls in total.'.format(len(urls)))
    main(urls, opinionFile)
