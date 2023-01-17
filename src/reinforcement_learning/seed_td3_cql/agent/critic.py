import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_Critic, self).__init__()

		net_width = 2 * (state_dim + action_dim)
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, net_width)  #没有先提取特征
		self.l2 = nn.Linear(net_width, net_width)
		self.l3 = nn.Linear(net_width, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, net_width)
		self.l5 = nn.Linear(net_width, net_width)
		self.l6 = nn.Linear(net_width, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2