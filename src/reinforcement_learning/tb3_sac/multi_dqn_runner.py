from concurrent.futures import process
import torch.multiprocessing as mp
from dqn_runner import train
from agent.agent_dqn import DQN

if __name__ == '__main__':
    num_processes = 4
    model = DQN()
    model.shared_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in process:
        p.join()