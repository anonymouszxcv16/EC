import copy
import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

# Layer normalization.
def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class ReplayMemory:
    def __init__(self, capacity, args):
        self.capacity = capacity
        self.args = args
        self.ptr = 0
        self.size = 0

        # Allocate tensor buffers
        self.state = torch.zeros((capacity, *args.state_shape), dtype=torch.float32, device=args.device)
        self.action = torch.zeros((capacity, *args.action_shape), dtype=torch.long, device=args.device)
        self.next_state = torch.zeros((capacity, *args.state_shape), dtype=torch.float32, device=args.device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32, device=args.device)
        self.correction = torch.zeros((capacity, 1)).to(args.device)
        self.zsa = torch.zeros((capacity, args.zs_dim)).to(args.device)
        self.done = torch.zeros((capacity, 1), dtype=torch.bool, device=args.device)

        self.priority = torch.ones((capacity,), dtype=torch.float32, device=args.device)

    # Select tuples whose rewards does not fit to their immediate rewards
    def select(self):
        # topk correction priorities
        idxs = self.priority.topk(k=min(self.args.k, self.size), largest=True)[1]

        # topk_NN closest state
        sim_idxs = [(self.state[idx] - torch.cat((self.state[:idx],
                                                self.state[idx + 1:self.size]))).abs().mean(dim=1).topk(k=self.args.k_NN,
                                                                                                      largest=False)[1]
                    for idx in idxs][0]

        # avg topk_NN rewards
        to_correct = (self.state[idxs], self.action[idxs], self.next_state[idxs], self.reward[idxs],
                      torch.tensor([(self.reward[sim_idx]) for sim_idx in sim_idxs]).mean(),
                      self.done[idxs])
        self.remove(idxs)

        return to_correct

    def remove(self, idxs):
        for idx in idxs:
            for item in [self.state, self.action, self.next_state, self.reward, self.zsa, self.done]:
                item[idx:, :] = torch.cat((item[idx + 1:, :], torch.zeros((1, item.shape[1])).to(self.args.device)))

        self.size -= len(idxs)

    # Fit all rewards of tuples up to <fit_max> tuples according the conscious replay value reward ratio
    def fit(self):
        idxs = range(self.size)
        self.reward[:self.size, 0] = self.correction[idxs, 0]

        sample = self.sample()
        self.remove(idxs)

        return sample

    # Add tuple.
    def push(self, state, action, next_state, reward, correction, zsa, done):
        self.state[self.ptr:self.ptr + state.shape[0]] = state
        self.action[self.ptr:self.ptr + state.shape[0]] = action
        self.next_state[self.ptr:self.ptr + state.shape[0]] = next_state
        self.reward[self.ptr:self.ptr + state.shape[0]] = reward
        self.correction[self.ptr:self.ptr + state.shape[0]] = correction
        self.zsa[self.ptr:self.ptr + state.shape[0]] = zsa
        self.done[self.ptr:self.ptr + state.shape[0]] = done

        self.priority[self.ptr:self.ptr + state.shape[0]] = self.priority.max().item() if self.size > 0 else 1.0  # max priority

        self.ptr = (self.ptr + state.shape[0]) % self.capacity
        self.size = min(self.size + state.shape[0], self.capacity)

    def sample(self, priority=False):
        if priority:
            probs = self.priority[:self.size]
            probs /= probs.sum()

            idxs = torch.multinomial(probs, self.args.batch_size, replacement=True)
        else:
            idxs = np.random.randint(0, self.size, size=min(self.args.batch_size, self.size))

        batch = {
            'state': self.state[idxs],
            'action': self.action[idxs],
            'next_state': self.next_state[idxs],
            'reward': self.reward[idxs],
            'zsa': self.zsa[idxs],
            'done': self.done[idxs],
            'idxs': idxs
        }

        return batch

    def update_priorities(self, idxs, td_errors):
        self.priority[idxs] = td_errors.squeeze().detach()

    # Load offline dataset.
    def load_D4RL(self, dataset, preprocessor):
        # Collect all transitions
        all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []

        for episode in dataset:
            # Process all observations through your preprocessor
            obs_processed = np.array([preprocessor.preprocess(frame) for frame in episode.observations])

            # Align transitions: s_t -> a_t -> r_t -> s_{t+1}
            states = obs_processed[:-1]  # (T, 4096)
            next_states = obs_processed[1:]  # (T, 4096)

            all_states.append(states)
            all_actions.append(episode.actions.astype(np.int64))
            all_rewards.append(episode.rewards)
            all_next_states.append(next_states)
            all_dones.append((episode.terminations | episode.truncations).astype(np.float32))

        # Convert to your exact tensor format
        self.state = torch.tensor(np.concatenate(all_states), dtype=torch.float32, device=self.args.device)
        self.action = torch.tensor(np.concatenate(all_actions), dtype=torch.long, device=self.args.device).unsqueeze(-1)
        self.next_state = torch.tensor(np.concatenate(all_next_states), dtype=torch.float32, device=self.args.device)
        self.reward = torch.tensor(np.concatenate(all_rewards).reshape(-1, 1), dtype=torch.float32, device=self.args.device)
        self.done = torch.tensor(np.concatenate(all_dones).reshape(-1, 1), dtype=torch.bool, device=self.args.device)
        self.size = self.state.shape[0]

    def __len__(self):
        return self.size

class Encoder(nn.Module):
    def __init__(self, args, zs_dim=256, hdim=256, activ=F.elu):
        super(Encoder, self).__init__()

        self.activ = activ

        # state encoder
        self.zs1 = nn.Linear(args.state_shape[0], hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + args.action_shape[0], hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

        self.args = args

    def zs(self, state):
        # Fully connected.
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))

        # Normalization.
        zs = AvgL1Norm(self.zs3(zs))

        return zs

    def zsa(self, zs, action):
        # Fully connected.
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)

        return zsa

class DQN(nn.Module):

    def __init__(self, args):
        super(DQN, self).__init__()

        self.layer1 = nn.ParameterList([nn.Linear(*args.state_shape, args.hdim) for _ in range(args.N)])
        self.layer2 = nn.ParameterList([nn.Linear(args.zs_dim + args.hdim, args.hdim) for _ in range(args.N)])
        self.layer3 = nn.ParameterList([nn.Linear(args.hdim, args.n_actions) for _ in range(args.N)])

        self.args = args

    # Called with either one element to determine next action, or a batch during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, zs):
        q_values = []

        for i in range(self.args.N):
            q = AvgL1Norm(self.layer1[i](x))

            q = torch.cat([q, zs], 1)

            q = F.relu(self.layer2[i](q))
            q_value = self.layer3[i](q)

            q_values.append(q_value)

        q_values = torch.stack(q_values, dim=0)
        return q_values


class Agent():
    def __init__(self, args, env, preprocessor):
        self.env = env
        self.args = args
        self.preprocessor = preprocessor

        self.t = 0

        self.policy_net = DQN(args).to(args.device)
        self.target_net = DQN(args).to(args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.encoder = Encoder(args).to(args.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=args.lr, amsgrad=True)

        self.memory = ReplayMemory(args.replay_size, args)
        self.unconscious_memory = ReplayMemory(args.replay_size, args)

        # q - q_target weights
        self.target_biases = []

        if "CQL" in self.args.policy:
            self.cql_alpha = torch.tensor(1.0, requires_grad=True, device=self.args.device)
            self.cql_alpha_optimizer = torch.optim.Adam([self.cql_alpha], lr=1e-4)

    def compute_cql_loss(self, state, zs, action, actor):
        with torch.no_grad():
            q_values = self.policy_net(state, zs).squeeze(2).gather(index=actor.reshape(1, state.shape[0], 1), dim=-1).flip(
                0).squeeze(-1)

            # Current Q-values (for data actions)
            y = self.policy_net(state, zs).squeeze(2).gather(index=action.reshape(1, state.shape[0], 1), dim=-1).flip(
                0).squeeze(-1)

        # Q values actor log sum exp.
        penalty_target = torch.logsumexp(q_values, dim=1, keepdim=True)
        cql_loss = (penalty_target - y).mean()

        return cql_loss

    def estimate_pser(self, td_loss, state):
        max_value = td_loss[state]

        for i in range(self.args.W):
            max_value = max(td_loss[state], td_loss[min(state + i, td_loss.shape[0] - 1)] * self.args.rho ** i)

        return max_value

    def get_epsilon(self):
        return self.args.eps_end + (self.args.eps_start - self.args.eps_end) * math.exp(-1. * self.t / self.args.eps_decay)

    def select_action(self, state, q_idx=0, greedy=False):
        sample = random.random()
        eps_threshold = self.get_epsilon()

        if sample > eps_threshold or greedy:
            with torch.no_grad():
                zs = self.fixed_encoder.zs(state)
                # t.max(1) will return the largest column value of each row.
                q_values = self.policy_net(state, zs)[q_idx]

                action = q_values.argmax(-1)
                return torch.tensor(action, dtype=torch.long).unsqueeze(1).to(self.args.device)
        else:

            return torch.tensor([[self.env.action_space.sample()]], device=self.args.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.args.batch_size:
            return

        # Sampling a batch
        batch = self.memory.sample(priority="PER" in self.args.policy or "RELO" in self.args.policy or "PSER" in self.args.policy)

        # Mask for non-final states
        non_final_mask = torch.tensor(~batch['done'].squeeze(-1), dtype=torch.bool, device=self.args.device) # shape [B]

        non_final_next_states = batch['next_state'][non_final_mask]
        state = batch['state']
        action = batch['action']
        reward = batch['reward']

        if "SALE" in self.args.policy:
            with torch.no_grad():
                next_zs = self.encoder.zs(batch['next_state'])

            zs = self.encoder.zs(state)
            pred_zs = self.encoder.zsa(zs, action)

            # Loss.
            encoder_loss = F.mse_loss(pred_zs, next_zs)
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward()
            self.encoder_optimizer.step()

        # Q-values
        with torch.no_grad():
            zs = self.fixed_encoder.zs(state)
            zsa = self.fixed_encoder.zsa(zs, action)

            # Update Replay Embeddings
            self.memory.zsa[batch['idxs']] = zsa

        q = self.policy_net(state, zs).squeeze(2).gather(index=action.reshape(1, state.shape[0], 1), dim=-1).flip(0).squeeze(-1)
        q_next = torch.zeros(self.args.N, state.shape[0], device=self.args.device, dtype=torch.long)

        with torch.no_grad():
            zs_target = self.fixed_encoder_target.zs(non_final_next_states)
            q_next[:, non_final_mask] = (self.target_net(non_final_next_states, zs_target).amax(-1).squeeze(-1).clone().detach().to(self.args.device,
                                                                                                                                    dtype=torch.long))

        # Target
        reward_batch = reward.reshape(1, reward.shape[0])
        q_next = (q_next * self.args.discount) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        td_loss = criterion(q, q_next)

        if self.args.offline == 1 and ("BC" in self.args.policy):
            actor = torch.tensor(self.policy_net(state, zs).argmax(-1), dtype=torch.float32).reshape((-1, 1))
            BC_loss = F.mse_loss(actor, action)
            td_loss += self.args.lmbda * q.abs().mean().detach() * BC_loss

            if "CQL" in self.args.policy:
                cql_loss = self.compute_cql_loss(state, zs, action, torch.tensor(actor, dtype=torch.int32))
                td_loss = .5 * td_loss + self.args.alpha_cql * cql_loss

        if "PER" in self.args.policy:
            td_error = (q - q_next).abs().squeeze(0)
            priority = td_error.pow(self.args.alpha)
            self.memory.update_priorities(batch['idxs'], priority)

        elif "PSER" in self.args.policy:
            td_error = (q - q_next).abs().squeeze(0)
            priority = torch.tensor([self.estimate_pser(td_error, i) for i in range(td_error.shape[0])]).to(self.args.device)
            priority = priority.pow(self.args.alpha)

            self.memory.update_priorities(batch['idxs'], priority)

        elif "RELO" in self.args.policy or "MC" in self.args.policy:
            with torch.no_grad():
                zs_target = self.fixed_encoder_target.zs(state)
                q_target = self.target_net(state, zs_target).squeeze(2).gather(index=action.reshape(1, state.shape[0], 1), dim=-1).flip(0).squeeze(-1).to(self.args.device,
                                                                                                                                                          dtype=torch.long)
                loss = (q - q_target).abs().squeeze(0)

            priority = loss.pow(self.args.alpha)
            self.memory.update_priorities(batch['idxs'], priority)

        # Optimize the model
        self.optimizer.zero_grad()
        td_loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if "MC" in self.args.policy:
            if self.t % self.args.interval == 0 and self.unconscious_memory.size > 0:
                state, action, next_state, reward, correction, done = self.memory.select()
                self.unconscious_memory.push(state, action, next_state, reward, correction, 0, done)

            if self.t % self.args.interval == self.args.interval // 2 and self.unconscious_memory.size > 0:
                batch = self.unconscious_memory.fit()
                self.memory.push(batch['state'], batch['action'], batch['next_state'], batch['reward'], batch['reward'], batch['zsa'], batch['done'])