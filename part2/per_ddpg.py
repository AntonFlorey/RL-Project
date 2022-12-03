import sys, os
from turtle import done

from matplotlib.pyplot import acorr
sys.path.insert(0, os.path.abspath(".."))
import copy
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from common import helper as h
from common.buffer import Batch, ReplayBuffer, PrioritizedReplayBuffer, PrioritizedReplayBuffer2

import time

expl_stddev = 0.1 #0.08

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x) # output shape [batch, 1]


class DDPG(object):
    def __init__(self, state_shape, action_dim, max_action, lr, gamma, tau, batch_size, alpha, beta, buffer_size=1e6, actor_hidden_dim=256, critic_hidden_dim=256,
                 initial_samples=500, num_dists=100, sort_interval=1000):
        state_dim = state_shape[0]
        self.action_dim = action_dim
        self.max_action = max_action
        self.pi = Policy(state_dim, action_dim, max_action, actor_hidden_dim).to(device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr)

        self.q = Critic(state_dim, action_dim, critic_hidden_dim).to(device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(state_shape, action_dim, batch_size, max_size=int(buffer_size), alpha=alpha, beta=beta, sort_interval=sort_interval, 
                                               sample_start=initial_samples, num_dists=num_dists)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = initial_samples # collect random data for better exploration
    
    def adjust_learning_rate(self, factor = 0.1):
        for g in self.pi_optim.param_groups:
            g['lr'] *= factor
        for g in self.q_optim.param_groups:
            g['lr'] *= factor

    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head # update the network once per transiton

        if self.buffer_ptr > self.random_transition: # update once have enough data
            for _ in range(update_iter):
                info = self._update()
        
        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
        return info

    def _update(self,):
        start_t = time.time()
        sampling_time = 0
        prio_update_time = 0

        a = time.time()
        batch:Batch = self.buffer.sample(device=device)
        sampling_time = time.time() - a

        # Task 2
        ########## Your code starts here. ##########
        # Hints: 1. compute the Q target with the q_target and pi_target networks
        #        2. compute the critic loss and update the q's parameters
        #        3. compute actor loss and update the pi's parameters
        #        4. update the target q and pi using h.soft_update_params() (See the DQN code)
        
        done_mask = batch.not_done.to(torch.int64)
        q_targ = batch.reward + torch.mul(self.gamma * self.q_target(batch.next_state, self.pi_target(batch.next_state)), done_mask)
        q_diff = q_targ.detach() - self.q(batch.state, batch.action)

        # debug importance weight
        #critic_loss = torch.mean(torch.mul(batch.extra["importance_weight"].detach(), torch.square(q_diff).flatten())) # importance sampling

        loss_fct = torch.nn.MSELoss()
        critic_loss = loss_fct(self.q(batch.state, batch.action), q_targ.detach())

        self.q_optim.zero_grad()
        critic_loss.backward()
        # update qs parameters
        self.q_optim.step()

        #actor loss 
        actor_loss = -torch.mean(self.q(batch.state, self.pi(batch.state)))
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        # target updates
        h.soft_update_params(self.q, self.q_target, self.tau)
        h.soft_update_params(self.pi, self.pi_target, self.tau)

        # update replay priorities 
        q_priority = torch.abs(q_diff).detach().flatten().tolist()
        a = time.time()
        self.buffer.update_priorities(q_priority, batch.extra["buffer_id"])
        prio_update_time = time.time() - a
        # print("Sampling Overhead: ", sampling_time / (time.time() - start_t))
        # print("Prio Update Overhead: ", prio_update_time / (time.time() - start_t))
        ########## Your code ends here. ##########

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {}


    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(device)

        if (not evaluation) and self.buffer_ptr < self.random_transition: # collect random trajectories for better exploration.
            action = torch.rand(self.action_dim)
        else:
            expl_noise = expl_stddev * self.max_action # the stddev of the expl_noise if not evaluation
            
            # Task 2
            ########## Your code starts here. ##########
            # Use the policy to calculate the action to execute
            # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
            # Hint: Make sure the returned action's shape is correct.
            # pass

            action = self.pi(x)

            if (not evaluation):
                action += np.random.normal(0.0, expl_noise)

            ########## Your code ends here. ##########

        return action, {} # just return a positional value


    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)

    
    # You can implement these if needed, following the previous exercises.
    def load(self, filepath):
        self.pi.load_state_dict(torch.load(filepath))
        self.pi_target.load_state_dict(torch.load(filepath))
    
    def save(self, filepath):
        torch.save(self.pi.state_dict(), filepath)