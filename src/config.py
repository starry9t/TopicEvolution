import os
import sys
import torch
from configparser import SafeConfigParser
from Utils.utils import trace, check_path


def get_correct_args(config, items, section):
    d = {}
    for key, value in items:
        val = None
        try:
            val = config[section].getint(key)
        except:
            try:
                val = config[section].getfloat(key)
            except:
                try:
                    val = config[section].getboolean(key)
                except:
                    val = value
        d[key]= val 
    return d 


def read_config(args, args_parser, config_file=None):
    
   
#    if config_file is None:
#        return args
#    if not os.path.isfile(config_file):
#        trace("""# Cannot find the configuration file. 
#            {} does not exist! Please check.""".format(config_file))
#        sys.exit(1)
#        
#    config = SafeConfigParser()
#    config.read(config_file)
#    for section in config.sections():
#        
#        default = get_correct_args(config, config.items(section), section)
#        
#        temp_d = {
#            k:v for k,v in filter(
#                lambda x: hasattr(args, x[0]), default.items())}
#        
#        args_parser.set_defaults(**temp_d)
        
        
    args = args_parser.parse_args()
    
    # GPU
    if args.useGPU:
        if not torch.cuda.is_available():
            args.useGPU = False
            print('no GPU available.')
        else:
            print('use GPU')
    else:
        print('do NOT use GPU.')
    args.device = torch.device("cuda:{}".format(torch.cuda.current_device()) if args.useGPU else "cpu")
    
    # path 
    project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    args.project_path = os.path.abspath(project_path)
    args.data_path = os.path.join(project_path, 'data/{}'.format(args.dataname))
    args.output_path = os.path.join(project_path, '{}/{}'.format(args.odir, args.flag))
    args.log_file = os.path.join(args.output_path, 'log_{}.txt'.format(args.flag))
    #args.save_vocab = os.path.join(project_path, 'save/vocab') 
    args.train_file = os.path.join(args.data_path, '{}.txt'.format(args.filename))
    #args.dev_file = os.path.join(args.data_path, 'dev.txt')
    #args.test_file = os.path.join(args.data_path, 'test.txt')
    args.topic_file = os.path.join(args.output_path, 'topics_{}.txt'.format(args.flag))
    args.topic_evo_file = os.path.join(args.output_path, 'topics_evo_{}.txt'.format(args.flag))
    args.tws_file = os.path.join(args.output_path, 'tws.pkl')
    check_path(args.project_path, 'report')
    check_path(args.data_path, 'report')
    check_path(args.output_path, 'add')
    #check_path(args.log_file, 'add')
    check_path(args.train_file, 'report')
    
    
    
    return args


