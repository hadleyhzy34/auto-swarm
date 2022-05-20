from easydict import EasyDict as edict
import torch

Config = edict()

Config.num_processes = 8
Config.namespace = 'tb3_'
Config.Train = edict()
Config.Test = edict()

# training parameters
Config.Train.batch_size = 64
Config.Train.state_size = 364
Config.Train.action_size = 5
Config.Train.lr = 1e-3
# Config.Train.device = torch.device('cuda')
Config.Train.device = torch.device('cpu')
Config.Train.episodes = 6000

