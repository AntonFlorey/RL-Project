import sys, os
sys.path.insert(0, os.path.abspath(".."))
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np
from common import helper as h
from collections import namedtuple
from common.helper import discount_rewards

PPOBatch = namedtuple('Batch', ['state', 'action', 'action_logprob', 'next_state', 'reward', 'dis_rew', 'gae', 'done'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Actor-critic agent
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hd: int):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, hd)), nn.Tanh(),
            layer_init(nn.Linear(hd, hd)), nn.Tanh(),
            layer_init(nn.Linear(hd, action_dim), std=0.01),
        )
        # Task 1: Implement actor_logstd as a learnable parameter
        # Use log of std to make sure std doesn't become negative during training
        self.actor_logstd = nn.parameter.Parameter(torch.zeros(action_dim, device=device), requires_grad=True)

    def forward(self, state):
        # Get mean of a Normal distribution (the output of the neural network)
        action_mean = self.actor_mean(state)

        # Make sure action_logstd matches dimension of action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)

        # Exponentiate the log std to get actual std
        action_std = torch.exp(action_logstd)

        # Task 1: Create a Normal distribution with mean of 'action_mean' and standard deviation of 'action_logstd', and return the distribution
        probs = Normal(action_mean, action_std)

        return probs

class Value(nn.Module):
    def __init__(self, state_dim, hd: int):
        super().__init__()
        self.value = nn.Sequential(
            layer_init(nn.Linear(state_dim, hd)), nn.Tanh(),
            layer_init(nn.Linear(hd, hd)), nn.Tanh(),
            layer_init(nn.Linear(hd, 1)))
    
    def forward(self, x):
        return self.value(x).squeeze(1) # output shape [batch,]


class PPO(object):
    def __init__(self, state_dim, action_dim, lr, gamma, policy_hd:int=64, value_hd:int=64,
                 clip_eps=0.2, gae_lambda=0.95, entropy_weight=0.01, num_minibatches=15, minibatch_size=256, max_grad=None):
        self.policy = Policy(state_dim, action_dim, policy_hd).to(device)
        self.value = Value(state_dim, value_hd).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr,)

        self.gamma = gamma

        # optimizer parameters
        self.max_grad_norm = max_grad

        # a simple buffer
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []

        # PPO parameters
        self.minibatch_size = minibatch_size
        self.batches_per_episode = num_minibatches
        self.entropy_weight = entropy_weight
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda

    def _gae(self, rewards, states, next_states, dones):
        values = self.value.forward(states)
        next_values = self.value.forward(next_states)

        deltas = [(r + (1.0 - d) * self.gamma * nv - v) for r, nv, v, d in zip(rewards, values, next_values, dones)]
        deltas = torch.stack(deltas).to(device).squeeze(-1)
        gaes = deltas.clone()

        for t in reversed(range(len(deltas - 1))):
            gaes[t] = gaes[t] + (1 - dones[t]) * (self.gae_lambda * self.gamma * gaes[t])

        return gaes

    def _update(self, batch: PPOBatch):
        new_dist = self.policy.forward(batch.state)
        new_value = self.value.forward(batch.state)

        # compute ratios
        prob_ratio = torch.exp(new_dist.log_prob(batch.action).sum(dim=-1) - batch.action_logprob.sum(dim=-1).detach())
        clipped_ratio = torch.clamp(prob_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        # scale gae and get value target
        value_target = batch.gae + new_value
        gae_scaled = (batch.gae - torch.mean(batch.gae)) / (torch.std(batch.gae) + 1e-8)

        # compute the advantage function
        adv = batch.dis_rew - new_value.detach()
        adv_scaled = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        # compute actor loss 
        l_clip = -torch.mean(torch.min(prob_ratio * gae_scaled.detach(), clipped_ratio * gae_scaled.detach()))
        l_entropy = -torch.mean(new_dist.entropy())
        actor_loss = l_clip + self.entropy_weight * l_entropy 

        # compute value loss
        value_loss_fct = torch.nn.MSELoss()
        value_loss = value_loss_fct(value_target.detach(), new_value)

        # optimize
        self.policy_optimizer.zero_grad()
        actor_loss.backward()

        self.value_optimizer.zero_grad()
        value_loss.backward()

        if self.max_grad_norm is not None:
            nn.utils.clip_grad.clip_grad_norm(self.policy.parameters(), self.max_grad_norm)
            nn.utils.clip_grad.clip_grad_norm(self.value.parameters(), self.max_grad_norm)
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        return {}

    def update(self,):
        actions = torch.stack(self.actions, dim=0).to(device).squeeze(-1)
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(device).squeeze(-1)
        dones = torch.stack(self.dones, dim=0).to(device).squeeze(-1)
        # clear buffer
        self.states, self.actions, self.action_probs, self.rewards, self.dones, self.next_states = [], [], [], [], [], []

        # compute discounted rewards
        disc = discount_rewards(rewards, self.gamma)

        # compute gae
        gaes = self._gae(rewards, states, next_states, dones)

        # sample and learn from minibatches TODO: put everything into one batch optionally
        for _ in range(self.batches_per_episode):
            ind = np.random.randint(low=0, high=actions.shape[0], size=self.minibatch_size)
            batch = PPOBatch(
                state=states[ind],
                action=actions[ind],
                action_logprob=action_probs[ind],
                next_state=next_states[ind],
                reward=rewards[ind],
                dis_rew=disc[ind],
                gae=gaes[ind],
                done=dones[ind]
            )
            self._update(batch)

        return {}

    def get_action(self, observation, evaluation=False):
        """Return action (np.ndarray) and logprob (torch.Tensor) of this action."""
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(device)

        # Task 1
        ########## Your code starts here. ##########
        # Hints: 1. the self.policy returns a normal distribution, check the PyTorch document to see 
        #           how to calculate the log_prob of an action and how to sample.
        #        2. if evaluation, return mean, otherwise, return a sample
        #        3. the returned action and the act_logprob should be the torch.Tensors.
        #            Please always make sure the shape of variables is as you expected.
        
        probs = self.policy.forward(x)
        action = 0

        if evaluation:
            action = probs.mean
        else: 
            action =  probs.sample()

        # clamp action (bipedalwalker specific)
        action = torch.clamp(action, -1.0, 1.0)
        act_logprob = probs.log_prob(action)
    
        ########## Your code ends here. ###########

        return action, act_logprob

    def record(self, observation, action, action_prob, reward, done, next_observation):
        self.states.append(torch.tensor(observation, dtype=torch.float32))
        self.actions.append(torch.tensor(action, dtype=torch.float32))
        self.action_probs.append(action_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.dones.append(torch.tensor([done], dtype=torch.float32))
        self.next_states.append(torch.tensor(next_observation, dtype=torch.float32))

    # You can implement these if needed, following the previous exercises.
    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath))
    
    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)