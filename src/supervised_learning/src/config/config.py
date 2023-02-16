from easydict import EasyDict as edict
import torch

Config = edict()

Config.weight_file = "/home/fdrobot/development/ssrpp/data/weight/cur.pth"
Config.weight_folder = "/home/fdrobot/development/ssrpp/data/weight/"
Config.data_file = "/home/fdrobot/development/ssrpp/data/csv"
Config.tgt_file = "/home/fdrobot/development/ssrpp/data/pt"
Config.file_list = "/home/fdrobot/development/ssrpp/data/file_list.csv"
Config.length_list = "/home/fdrobot/development/ssrpp/data/length_list.csv"

# Config.data_file = "/home/fdrobot/development/ssrpp/data/pt"

Config.num_processes = 8
Config.namespace = 'tb3_'

# model hyper-parameters
Config.Model = edict()
Config.Model.length = 15


# shared memory
Config.Memory = edict()
Config.Memory.capacity = 2000

# training parameters
Config.Train = edict()
Config.Train.batch_size = 512
Config.Train.state_dim = 364
Config.Train.action_dim = 2
Config.Train.lr = 1e-3
Config.Train.device = torch.device('cuda')
# Config.Train.learner_device = torch.device('cuda')
Config.Train.episodes = 30
Config.Train.update_global_iter = 10

# testing parameters
Config.Test = edict()
Config.Test.batch_size = 1
Config.Test.state_dim = 364
Config.Test.action_dim = 2
Config.Test.device = torch.device('cuda')

# evaluation parameters
Config.Eval = edict()
Config.Eval.batch_size = 1
Config.Eval.device = torch.device('cpu')

