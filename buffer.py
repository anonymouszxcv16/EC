import random
import numpy as np
import torch


# Replay buffer
class LAP(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		zsa_dim,
		device,
		args,
		max_size=1e6,
		batch_size=256,
		max_action=1,
		normalize_actions=True,
		prioritized=True
	):

		# Parameters.
		max_size = int(max_size)
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.device = device
		self.batch_size = batch_size

		self.action_dim = action_dim
		self.state_dim = state_dim
		self.zsa_dim = zsa_dim

		# Memory
		self.state = torch.zeros((max_size, state_dim)).to(args.device)
		self.action = torch.zeros((max_size, action_dim)).to(args.device)

		# ppo
		self.log_policy = torch.zeros((max_size, 1)).to(args.device)

		self.next_state = torch.zeros((max_size, state_dim)).to(args.device)
		self.reward = torch.zeros((max_size, 1)).to(args.device)

		# mc score
		self.mc_score = torch.zeros((max_size, 1)).to(args.device)

		# immediate reward correction
		self.r_correction = torch.zeros((max_size, 1)).to(args.device)

		# monte carlo return correction
		self.mc_correction = torch.zeros((max_size, 1)).to(args.device)

		self.zsa = torch.zeros((max_size, zsa_dim)).to(args.device)
		self.not_done = torch.zeros((max_size, 1)).to(args.device)

		self.prioritized = prioritized

		if prioritized:
			self.priority = torch.zeros(max_size, device=device)
			self.prioritized = True
			self.max_priority = 1

		self.normalize_actions = max_action if normalize_actions else 1

		self.args = args

	# Select tuples whose rewards does not fit to their immediate rewards
	def select(self):
		# topk correction priorities
		idxs = self.priority.topk(k=min(self.args.k, self.size), largest=True)[1]

		# topk_NN closest ZSA
		sim_idxs = [(self.zsa[idx] - torch.cat((self.zsa[:idx],
												self.zsa[idx + 1:self.size]))).abs().mean(dim=1).topk(k=self.args.k_NN, largest=False)[1]
					for idx in idxs][0]

		# avg topk_NN rewards
		to_correct = (self.state[idxs], self.action[idxs], self.next_state[idxs], self.reward[idxs],

					  # immediate reward average
					  torch.tensor([(self.reward[sim_idx]) for sim_idx in sim_idxs]).mean(),

					  # monte carlo return average
					  torch.tensor([(self.mc_score[sim_idx]) for sim_idx in sim_idxs]).mean(),

					  self.not_done[idxs])
		self.remove(idxs)

		return to_correct

	def remove(self, idxs):
		for idx in idxs:
			# all entries
			for item in [self.state, self.action, self.next_state, self.reward, self.mc_score, self.zsa, self.not_done]:
				item[idx:, :] = torch.cat((item[idx + 1:, :], torch.zeros((1, item.shape[1])).to(self.device)))

		self.size -= len(idxs)

	# Fit all rewards of tuples up to <fit_max> tuples according the conscious replay value reward ratio
	def fit(self):
		idxs = range(self.size)
		self.reward[:self.size, 0] = self.r_correction[idxs, 0]
		self.mc_score[:self.size, 0] = self.mc_correction[idxs, 0]

		sample = self.sample(ind=torch.range(0, self.size, dtype=torch.int64, device=self.args.device))

		self.remove(idxs)

		return sample

	# Add tuple.
	def add(self, state, action, next_state, reward, r_correction, mc_correction, zsa, done, mc_score=None):
		self.state[self.ptr:self.ptr + state.shape[0]] = state
		self.action[self.ptr:self.ptr + state.shape[0]] = action / self.normalize_actions
		self.next_state[self.ptr:self.ptr + state.shape[0]] = next_state
		self.reward[self.ptr:self.ptr + state.shape[0]] = reward

		self.mc_correction[self.ptr:self.ptr + state.shape[0]] = mc_correction
		self.r_correction[self.ptr:self.ptr + state.shape[0]] = r_correction

		self.zsa[self.ptr:self.ptr + state.shape[0]] = zsa
		self.not_done[self.ptr:self.ptr + state.shape[0]] = 1. - done

		if mc_score is not None:
			self.mc_score[self.ptr:self.ptr + state.shape[0]] = mc_score
		
		if self.prioritized:
			self.priority[self.ptr:self.ptr + state.shape[0]] = self.max_priority

		self.ptr = (self.ptr + state.shape[0]) % self.max_size
		self.size = min(self.size + state.shape[0], self.max_size)

	# Sample tuple.
	def sample(self, prioritized=False, ind=None):
		if ind is not None:
			self.ind = ind

		elif "REINFORCE" in self.args.policy:
			# trajectory
			start = random.randint(0, self.size - 1 - self.batch_size)
			end = start + self.batch_size - 1

			self.ind = torch.range(start, end, dtype=torch.int64)
		elif prioritized:
			csum = torch.cumsum(self.priority[:self.size], 0)
			val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
			self.ind = torch.searchsorted(csum, val).cpu().data.numpy()

		else:
			self.ind = np.random.randint(0, self.size, size=min(self.batch_size, self.size), dtype=np.int64)

		return (
			torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),

			torch.tensor(self.log_policy[self.ind], dtype=torch.float, device=self.device),

			torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),

			torch.tensor(self.mc_score[self.ind], dtype=torch.float, device=self.device),

			torch.tensor(self.zsa[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device)
		)

	def update_priority(self, priority):
		self.priority[self.ind] = priority.reshape(-1).detach()
		self.max_priority = max(float(priority.max()), self.max_priority)

	def reset_max_priority(self):
		self.max_priority = float(self.priority[:self.size].max())

	# Load offline dataset.
	def load_D4RL(self, dataset):
		self.state = torch.tensor(dataset['observations'], dtype=torch.float, device=self.device)
		self.action = torch.tensor(dataset['actions'], dtype=torch.float, device=self.device)
		self.next_state = torch.tensor(dataset['next_observations'], dtype=torch.float, device=self.device)
		self.reward = torch.tensor(dataset['rewards'].reshape(-1, 1), dtype=torch.float, device=self.device)
		self.not_done = torch.tensor(1. - dataset['terminals'].reshape(-1, 1), dtype=torch.float, device=self.device)
		self.size = self.state.shape[0]

		if self.prioritized:
			self.priority = torch.ones(self.size).to(self.device)