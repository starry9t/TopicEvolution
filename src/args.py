#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:44:56 2019

@author: Yu Zhou
"""

from argparse import ArgumentParser

def parse_args():
#def parse_args(model_name, script_name): 
    parser = ArgumentParser()
    
    parse_base_args(parser)
    parse_data_args(parser)
    parse_train_args(parser)   
    
    subparsers = parser.add_subparsers(help='sub-command help')
    
    # Model Specific
    parse_lda(subparsers)
    parse_dtm(subparsers)
    parse_vae(subparsers)
    parse_sch(subparsers)
    parse_ge(subparsers)
#    parse_pygcn(subparsers)
    #parse_line(subparsers)

    args = parser.parse_args()
    return args, parser
    
########### Common
    
def parse_base_args(parser):
    group = parser.add_argument_group('Base')
    #parser.add_argument('--config', type=str, default='config/nmt.ini', help='path to datasets')
    parser.add_argument('--model', type=str, choices=['lda','dtm','vrnn_vmf','vrnn','vae','scholar','ge'], help='use which model')
    parser.add_argument('--dataname', type=str, default='arxiv', help='name of dataset')
    parser.add_argument('--filename', type=str, default='LG')
    parser.add_argument('--flag', type=str, default='flag', help='to differ from')
    parser.add_argument('--continuous', default=False, action='store_true',
                      help='whether to convey info into next timestep')
    
    # GPU
    group = parser.add_argument_group('GPU')
    group.add_argument('--useGPU', default=True)#, action='store_true')
    group.add_argument('--device', default=None,
                      help='GPU to use: default=%default')    
    
    # Directory and file path
    group = parser.add_argument_group('Path')
    group.add_argument('--project_path', type=str, default="")
    group.add_argument('--data_path', type=str, default="")
    group.add_argument('--output_path', type=str, default="")
    group.add_argument('--odir', type=str, default='output')
    
    group.add_argument('--log_file', type=str, default="")
    group.add_argument('--train_file', type=str, default="")
    #group.add_argument('--dev_file', type=str)
    #group.add_argument('--test_file', type=str)
    group.add_argument('--topic_file', type=str, default="")
    group.add_argument('--topic_evo_file', type=str, default="")
    group.add_argument('--tws_file', type=str, default="")
    
    
def parse_data_args(parser):
    group = parser.add_argument_group('Data')
    group.add_argument('--unit', type=str, default='year', help='unit which cutting data according to')
    group.add_argument('-tp','--time_period', type=int, default=1, help='number which cutting data by')
    group.add_argument('--min_num_docs', type=int, default=100, 
                       help='throw it if number of documents in a time period is less than')
    group.add_argument('--min_num_words', type=int, default=30, 
                       help='throw it if number of words in document is less than')
    group.add_argument('--deli', type=str, default='uni', choices=['uni','phrase'])    
    
#    group.add_argument('-vm','--vocab_method', type=tuple, default=('tokp',2000))    
    group.add_argument('-vm','--vocab_method', type=str, default='tfidf', choices=['topk','all','highlow','tfidf'], 
                       help='number which cutting data by')
    
    group.add_argument('-rv','--reserve_vocab', type=int, default=2000, help='reserved vocab size if vm is topk')
    group.add_argument('-hlh','--highlow_high',type=float, default=0.1, help='percent of removed vocab if vm is highlow')
    group.add_argument('-hll','--highlow_low',type=float, default=0.3, help='percent of removed vocab if vm is highlow')
    
    #group.add_argument('--max_seq_len', type=int, default=20, help='Maximum sequence length')
    #group.add_argument('--save_vocab', type=str, help='path to vocab files')
    
    # tbc    
    
def parse_train_args(parser):
    group = parser.add_argument_group('Train')
    group.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    group.add_argument('--print_every', type=int, default=2)
    group.add_argument('--save_every', type=int, default=1)  
    
    group.add_argument('--batch_size', type=int, default=64, help='Maximum batch size for training')   
    

    group.add_argument('--mask', default=False, action='store_true')
    group.add_argument('--early_stop', default=True, action='store_true')

#    group.add_argument('--param_init', type=float, default=0.1,
#                       help="""Parameters are initialized over uniform distribution with support (-param_init, param_init).
#                       Use 0 to not use initialization""")
#    group.add_argument('--param_init_glorot', action='store_true', help='Init parameters with xavier_uniform. Required for transformer')
    
############ Model Specific
    
def parse_lda(subparsers):
    parser_lda = subparsers.add_parser('lda', help='a help')  
    parser_lda.add_argument('--z_dim', type=int, default=16, help='Topic number')    

def parse_dtm(subparsers):
    parser_lda = subparsers.add_parser('dtm', help='a help')
    parser_lda.add_argument('--z_dim', type=int, default=16, help='Topic number')  
    
def parse_vae(subparsers):
    # subparsers for vae
    parser_vae = subparsers.add_parser('vae', help='a help')
    parser_vae.add_argument('--vrnn', default=False, action='store_true')
    parser_vae.add_argument('-ie', '--init_embedding', default=False, action='store_true',
                            help='default not to use embedding to initialize weights')
    parser_vae.add_argument('--h_dim', type=int, default=100, help='Dimension of hidden layers')
    parser_vae.add_argument('--z_dim', type=int, default=20, help='Topic number')
    parser_vae.add_argument('--sampler', type=str, default='gaussian')  
    parser_vae.add_argument('--kappa', type=int, default=1)
    parser_vae.add_argument('--initial_mu', type=float, default=0.0)
    parser_vae.add_argument('--initial_sigma', type=float, default=1e-1)
    parser_vae.add_argument('-lr', type=float, default=0.0001, help='bar help')
    
    parser_vae.add_argument('--sparseW', default=0.0, type=float)
    parser_vae.add_argument('--entMax', default=False, action='store_true')
    
    
    #parser_vae.add_argument('--h_layers', type=int, default=1, help='Size of rnn hidden states')
    #parser_vae.add_argument('--bias', type=bool, default=False)
      
def parse_sch(subparsers):
    # subparsers for scholar
    parser_sch = subparsers.add_parser('sch', help='a help')   
    #usage = "%prog input_dir"

    
    parser_sch.add_argument('--z_dim', type=int, default=20,
                      help='Size of latent representation (~num topics): default=%default')
    parser_sch.add_argument('--h_dim', type=int, default=100, help='Dimension of hidden layers')
    
    parser_sch.add_argument('--lr', type=float, default=0.002,
                      help='Initial learning rate: default=%default')
    parser_sch.add_argument('--mmt', dest='momentum', type=float, default=0.99,
                      help='beta1 for Adam: default=%default')
    
    parser_sch.add_argument('--emb_method', type=str, default='glove')
    parser_sch.add_argument('--no_bg', default=False, action='store_true',
                      help='whether to initialize background using overall word frequencies')    
    parser_sch.add_argument('--def_reg', action="store_true", default=False,
                      help='Use default regularization: default=%default')
    parser_sch.add_argument('--l1_topics', type=float, default=0.0,
                      help='Regularization strength on topic weights: default=%default')
    parser_sch.add_argument('--l1_topic_covars', type=float, default=0.0,
                      help='Regularization strength on topic covariate weights: default=%default')
    parser_sch.add_argument('--l1_interactions', type=float, default=0.0,
                      help='Regularization strength on topic covariate interaction weights: default=%default')    
 
    parser_sch.add_argument('--seed', type=int, default=None,
                      help='Random seed: default=%default') 
    
    parser_sch.add_argument('--alpha', type=float, default=1.0,
                      help='Hyperparameter for logistic normal prior: default=%default')
    parser_sch.add_argument('--adam_beta1', type=float, default=0.99)
    parser_sch.add_argument('--adam_beta2', type=float, default=0.999)    
      
#    parser_sch.add_argument('--interactions', action="store_true", default=False,
#                      help='Use interactions between topics and topic covariates: default=%default')
    
#    parser_sch.add_argument('--covars_predict', action="store_true", default=False,
#                      help='Use covariates as input to classifier: default=%default')
    
#    parser_sch.add_argument('--min-prior-covar-count', type=int, default=None,
#                      help='Drop prior covariates with less than this many non-zero values in the training dataa: default=%default')
#    parser_sch.add_argument('--min-topic-covar-count', type=int, default=None,
#                      help='Drop topic covariates with less than this many non-zero values in the training dataa: default=%default')
#
#
#    parser_sch.add_argument('--l2-prior-covars', type=float, default=0.0,
#                      help='Regularization strength on prior covariate weights: default=%default')
#    parser_sch.add_argument('-o', dest='output_dir', type=str, default='output',
#                      help='Output directory: default=%default')
#    parser_sch.add_argument('--emb-dim', type=int, default=300,
#                      help='Dimension of input embeddings: default=%default')
#    parser_sch.add_argument('--w2v', dest='word2vec_file', type=str, default=None,
#                      help='Use this word2vec .bin file to initialize and fix embeddings: default=%default')


#    parser_sch.add_argument('--dev-folds', type=int, default=0,
#                      help='Number of dev folds: default=%default')
#    parser_sch.add_argument('--dev-fold', type=int, default=0,
#                      help='Fold to use as dev (if dev_folds > 0): default=%default')
#    parser_sch.add_argument('--device', type=int, default=None,
#                      help='GPU to use: default=%default')
   
def parse_ge(subparsers):
    # subparsers for scholar
    parser_ge = subparsers.add_parser('graph_emb', help='a help')   
    
    parser_ge.add_argument('--graph', default='topicmap', help="graph model")
    parser_ge.add_argument('--cluster', type=str, default='kmeans', help='clustering method')
    parser_ge.add_argument('--init_way', default='standard', help='how to initialize centroids')
    
    parser_ge.add_argument('--z_dim', type=int, default=20,
                      help='Size of latent representation (~num topics): default=%default')
    parser_ge.add_argument('--h_dim', type=int, default=100,
                      help='Dimension of word embedding (~): default=%default')   

    parser_ge.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser_ge.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser_ge.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate.')
    parser_ge.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser_ge.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    
#def parse_pygcn(subparsers):
#    parser_pygcn = subparsers.add_parser('pygcn', help='a help') 
#
#    parser_pygcn.add_argument('--fastmode', action='store_true', default=False,
#                        help='Validate during training pass.')
#    parser_pygcn.add_argument('--seed', type=int, default=42, help='Random seed.')
#
#    parser_pygcn.add_argument('--lr', type=float, default=0.01,
#                        help='Initial learning rate.')
#    parser_pygcn.add_argument('--weight_decay', type=float, default=5e-4,
#                        help='Weight decay (L2 loss on parameters).')
#    parser_pygcn.add_argument('--dropout', type=float, default=0.5,
#                        help='Dropout rate (1 - keep probability).')
    

############ During Training
    
    
#def parse_network_args(parser):
#    group = parser.add_argument_group('Network')
#    group.add_argument('--enc_num_layers', type=int, default=1, help='Number of layers in the encoder')
#    group.add_argument('--dec_num_layers', type=int, default=1, help='Number of layers in the decoder')
#    group.add_argument('--bidirectional', action='store_true', help='bidirecional encoding for encoder')
#    group.add_argument('--hidden_size', type=int, default=1000, help='Size of rnn hidden states')
#    group.add_argument('--latent_size', type=int, default=500, help='Size of latent states')
#    group.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'], help='the gate type')
#    group.add_argument('--attn_type', type=str, default='general', choices=['dot', 'general', 'mlp'], help='the attention type')
#    group.add_argument('--meanpool', action='store_true', help='use the mean pooling of encoder outputs for VAE')
#    group.add_argument('--dropout', type=float, default=0.3, help='Dropout probability; applied in RNN stacks')
#    
#def parse_embed_args(parser):
#    group = parser.add_argument_group('Embedding')
#    group.add_argument('--min_freq', type=int, default=5, help='min frequency for the prepared data')
#    group.add_argument('--src_embed_dim', type=int, default=500, help='word embedding size for src.')
    
#def parse_optim_args(parser):
#    group = parser.add_argument_group('Optimizer')
#    group.add_argument('--optim', default='adam', choices=['sgd', 'adagrad', 'adadelta', 'adam', 'sparseadam'],
#                       help = 'Optimization method')
#    group.add_argument('--adagrad_accum_init', type=float, default=0.0, 
#                       help="""Initializes the accumulator values in adagrad. 
#                       Mirrors the initial_accumulator_value option 
#                       in the tensorflow adagrad(use 0.1 for their default)""")
#    group.add_argument('--max_grad_norm', type=float, default=5,
#                       help="""If the norm of the gradient vector exceeds this, 
#                       renormalize it to have the norm equal to max_grad_norm""")
#    group.add_argument('--o_lr', type=float, default=1.0,
#                       help="""Starting learning rate.
#                       Recommended settings: sdf=1, adagrad=0.1, adadelta=1, adam=0.001""")
#    group.add_argument('--lr_decay_rate', type=float, default=0.5,
#                       help="""If updata_learning_rate, decay learning rate by this much if 
#                       (i) perplexity does not decrease on the validation set or 
#                       (ii) epoch has gone past start_decay_at""")
#    group.add_argument('--start_decay_at', type=int, default=8, help="""Start decaying every epoch after and including this epoch""")
#    group.add_argument('--decay_method', type=str, default="", choices=['noam'], help='Use a custom decay rate')
#    group.add_argument('--warmup_steps', type=int, default=4000, help="""Number of warmup steps for custom decay""")
#    group.add_argument('--alpha', type=float, default=0.9, help='The alpha parameter used by RMSprop')
#    group.add_argument('--eps', type=float, default=1e-8, help='The eps parameter used by RMSprop/Adam')
#    group.add_argument('--weight_decay', type=float, default=0, help='The weight_decay parameter used by RMSprop')
#    group.add_argument('--momentum', type=float, default=0, help='The momentum parameter used by RMSprop[0]/SGD[0.9]')
#    group.add_argument('--adam_beta1', type=float, default=0.9,
#                       help="""The beta1 parameter used by Adam.
#                       Almost without exception a value of 0.9 is used in the literature, seemingly giving good results,
#                       so we would discourage changing this value from the default without due consideration""")
#    group.add_argument('--adam_beta2', type=float, default=0.999,
#                       help="""The beta2 parameter used by Adam.
#                       Typically a value of 0.999 is recommended, as this is the value suggested by the original paper describing Adam,
#                       and is also the value adopted in other frameworks such as Tensorflow and Keras, i.e. see:
#                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
#                       https://keras.io/optimizers/ .
#                       Whereas recently the paper "Attention is All You Need" suggested a value of 0.98 for beta2, this parameter may
#                       not work well for normal models / default baselines.    """)
    
#def parse_topic_args(parser):
#    group = parser.add_argument_group('Topic Evolution')
#    group.add_argument('--output', default='pred.txt', 
#                       help="""Path to output the predictions(each line will be the topics""")
#    group.add_argument('--save_model', default='model', help="""Pre-trained models""")
#    group.add_argument('--beam_size', type=int, default=5, help='Beam size')
#    group.add_argument('--max_length', type=int, default=100, help='Maximum prediction length')
# 
#    
#
#    #### tbc
#    
#def parse_loss_args(parser):
#    group = parser.add_argument_group('Loss Functions')
#    group.add_argument('--kld_weight', default=0.05, type=float,
#                       help='weight for the Kullback-Leibler divergence Loss in total loss score')
#    group.add_argument('--start_increase_kld_at', default=8, type=int,
#                       help='start to increase KLD loss weight at.')
#    
#def parse_logging_args(parser):
#    group = parser.add_argument_group('Logging')
#    group.add_argument('--verbose', action='store_true', help='Print scores and predictions for each sentence') #tbc
#    group.add_argument('--plot_attn', action='store_true', help='Plot attention matrix for each pair')