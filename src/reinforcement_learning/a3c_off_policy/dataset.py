from numpy import genfromtxt
import os
import pdb
from pathlib import Path
import pandas as pd

directory = "/Users/hadley/Developments/self-supervised-recurrent-path-planning/data/"

file = "/home/hadley/Development/auto-swarm/src/reinforcement_learning/a3c_off_policy/tb3_0dataset.csv"
read_file = pd.read_csv (file)

my_data = genfromtxt(file, delimiter=',')

pdb.set_trace()

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        pdb.set_trace()
        my_data = genfromtxt(f,delimiter=';')
        
file_list = [f for f in directory.glob('**/*') if f.is_file()]

for file in os.listdir(directory):
    pdb.set_trace()
    my_data = genfromtxt(file[:-3]+'file.csv', delimiter=',')
