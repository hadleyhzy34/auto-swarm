from easydict import EasyDict as edict
import torch

Config = edict()

Config.num_processes = 8
Config.namespace = 'tb3_'
Config.Train = edict()
Config.Test = edict()

# shared memory
Config.Memory = edict()
Config.Memory.capacity = 2000

# training parameters
Config.Train.batch_size = 128
Config.Train.state_dim = 364
Config.Train.action_dim = 5
Config.Train.lr = 1e-3
# Config.Train.device = torch.device('cuda')
Config.Train.device = torch.device('cpu')
Config.Train.learner_device = torch.device('cuda')
Config.Train.episodes = 6000
