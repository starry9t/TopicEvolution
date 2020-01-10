# TopicEvolution
A python3 script to trace topic elution from time-ordered corpus.


## Installation

This code is written in python3 and has been tested with Ubuntu 14.0 x64, Anaconda Python 3.7 

#### Install Anaconda3
First download Anaconda3 installer from [here](https://www.anaconda.com/distribution/#linux) and run the following command in terminal:
```
bash Anaconda3-2018.12-Linux-x86_64.sh
```

#### Install other modules

```
$ pip install spacy
$ python -m spacy download en
$ pip install pyenchant 
$ pip install PrettyTable

$ conda install -c conda-forge gensim
$ conda install nltk

```
## Execution

If you have installed all dependencies successfully, it is very simple to rerun this project. 

#### Download dataset 
Just clone this repository to your machine and run pre_get_arxiv_data.py which might take a while to download our dataset.
```
$ git clone https://github.com/starry9t/TopicEvolution.git
$ cd TopicEvolution/
$ python pre_get_arxiv_data.py
```

#### Run main script
Simply run m_DTM.py by default you can get a simple result in default output folder(./output)
```
$ python m_DTM.py
```

For personally project setting, you can follow the instruction below to adjust parameters involved.

##### Parameters
- unit : make time slice by which unit, string type, default by 'year', can be choices in 'year', 'month', and 'day'

- tp : time period of the setting unit, int type, default by 1

- min_num_docs : throw the time period if the number of document in it is less than min_num_docs, int type, default by 100

- min_num_words : throw the document if the number of words in it is less than min_num_words, int type, default is 30
               
- vm : choices in 'topk', 'all', and 'highlow'
  - topk, reserve k words with highest frequency in data, if you set vm as topk, you will also need to set parameter '-reserve_vocab' which indicates the number K of vocabulary size which is default by 2000
  - all, by setting '-vm' as 'all', you will reserve all words from data
  - highlow, by setting '-vm' as 'highlow', you must also set parameters of '-highlow\_high' and '-highlow\_low' as these two number is the removed percentage of high and low frequency words 

- epochs : number of training iterations when training the model
- print_every : print losses on screen for every print_every epochs
- save_every : save losses in log.txt for every save_every epochs
- batch_size : maximum batch size for training
- early_stop : stop iteration in advance when the loss reach a particular threshold


After running successfully you will get result file in topic_evo.txt with content as below


![image](https://github.com/starry9t/TopicEvolution/blob/master/image/result_01.png)


![image](https://github.com/starry9t/TopicEvolution/blob/master/image/result_02.png)


![image](https://github.com/starry9t/TopicEvolution/blob/master/image/result_03.png)

## [Visualization](https://blpxspg.github.io/visualisation/index.html)

(

## File description

* data/ stores the input txt file

* rsc/ stores dependency scripts

* Output/ stores result text files

* pre_get_arxiv_data.py is the script to download data from Arxiv

* m_DTM.py is the main script to trace topic evolution and generate result files 

* post_result_analysis.py can show different analysis of the result such as coherence curves, word distribution maps etc.
