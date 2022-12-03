from collections import namedtuple
import numpy as np
import torch
import heapq
import functools
from common.prio_queue import BinaryHeap
import sys

Batch = namedtuple('Batch', ['state', 'action', 'next_state', 'reward', 'not_done', 'extra'])

class ReplayBuffer(object):
    def __init__(self, state_shape:tuple, action_dim: int, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        dtype = torch.uint8 if len(state_shape) == 3 else torch.float32 # unit8 is used to store images
        self.state = torch.zeros((max_size, *state_shape), dtype=dtype)
        self.action = torch.zeros((max_size, action_dim), dtype=dtype)
        self.next_state = torch.zeros((max_size, *state_shape), dtype=dtype)
        self.reward = torch.zeros((max_size, 1), dtype=dtype)
        self.not_done = torch.zeros((max_size, 1), dtype=dtype)
        self.extra = {}
    
    def _to_tensor(self, data, dtype=torch.float32):   
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def add(self, state, action, next_state, reward, done, extra:dict=None):
        self.state[self.ptr] = self._to_tensor(state, dtype=self.state.dtype)
        self.action[self.ptr] = self._to_tensor(action)
        self.next_state[self.ptr] = self._to_tensor(next_state, dtype=self.state.dtype)
        self.reward[self.ptr] = self._to_tensor(reward)
        self.not_done[self.ptr] = self._to_tensor(1. - done)

        if extra is not None:
            for key, value in extra.items():
                if key not in self.extra: # init buffer
                    self.extra[key] = torch.zeros((self.max_size, *value.shape), dtype=torch.float32)
                self.extra[key][self.ptr] = self._to_tensor(value)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device='cpu'):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.extra:
            extra = {key: value[ind].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[ind].to(device),
            action = self.action[ind].to(device), 
            next_state = self.next_state[ind].to(device), 
            reward = self.reward[ind].to(device), 
            not_done = self.not_done[ind].to(device), 
            extra = extra
        )
        return batch
    
    def get_all(self, device='cpu'):
        if self.extra:
            extra = {key: value[:self.size].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[:self.size].to(device),
            action = self.action[:self.size].to(device), 
            next_state = self.next_state[:self.size].to(device), 
            reward = self.reward[:self.size].to(device), 
            not_done = self.not_done[:self.size].to(device), 
            extra = extra
        )
        return batch


class PrioritizedReplayBuffer(object):
    def __init__(self, state_shape:tuple, action_dim: int, batch_size: int, max_size=int(1e6), 
                 alpha: float = 0.5, beta: float = 0.5, sort_interval: int = int(1e6), sample_start: int = 5000, num_dists: int = 100):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.sort_counter = 0

        dtype = torch.uint8 if len(state_shape) == 3 else torch.float32 # unit8 is used to store images
        self.state = torch.zeros((max_size, *state_shape), dtype=dtype)
        self.action = torch.zeros((max_size, action_dim), dtype=dtype)
        self.next_state = torch.zeros((max_size, *state_shape), dtype=dtype)
        self.reward = torch.zeros((max_size, 1), dtype=dtype)
        self.not_done = torch.zeros((max_size, 1), dtype=dtype)
        self.extra = {}

        self.alpha = alpha
        self.beta = beta
        self.sort_interval = sort_interval
        self.batch_size = batch_size
        self.sample_start = sample_start
        self.num_dist_parts = num_dists

        # pre compute buckets and probabilities
        self.dists = self.compute_dists()
        # priority queue
        self.priority_queue = BinaryHeap(priority_size=self.max_size)

    
    def compute_dists(self):

        res = {}
        dist_num = 1
        size_inc = int(np.floor(self.max_size / self.num_dist_parts))

        if size_inc > self.sample_start:
            print("SMALLEST BUFFER SIZE OPTION IS TO LARGE!")
            needed = np.ceil(self.max_size / self.sample_start)
            print("Need at least "+str(needed)+" buffer size options")

        # for each buffer size compute the buckets
        for buffersize in range(size_inc, self.max_size+1, size_inc):
            ps = [np.power(float(x), -self.alpha) for x in range(1, buffersize + 1)]
            den = np.sum(ps)
            pdf = np.array([x / den for x in ps])
            cdf = np.cumsum(pdf)
            buckets = [0] * (self.batch_size+1)
            buckets[self.batch_size] = buffersize # last bucket border is always the end of the buffer
            step = 1.0 / float(self.batch_size) # probability of each bucket region
            buffer_idx = 1
            for bucket_i in range(1, self.batch_size):
                while cdf[buffer_idx] < step:
                    buffer_idx += 1
                buckets[bucket_i] = buffer_idx
                buffer_idx += 1
                step += 1.0 / float(self.batch_size)
            distribution = {'pdf': pdf,
                            'buckets': buckets}
            res[dist_num] = distribution
            dist_num+= 1
            #print("dist "+str(dist_num-1)+" computed")

        return res

    def _to_tensor(self, data, dtype=torch.float32):   
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def add(self, state, action, next_state, reward, done, extra:dict=None):
        # add transition [s, a, s', r, d] to buffer
        self.state[self.ptr] = self._to_tensor(state, dtype=self.state.dtype)
        self.action[self.ptr] = self._to_tensor(action)
        self.next_state[self.ptr] = self._to_tensor(next_state, dtype=self.state.dtype)
        self.reward[self.ptr] = self._to_tensor(reward)
        self.not_done[self.ptr] = self._to_tensor(1. - done)

        if extra is not None:
            for key, value in extra.items():
                if key not in self.extra: # init buffer
                    self.extra[key] = torch.zeros((self.max_size, *value.shape), dtype=torch.float32)
                self.extra[key][self.ptr] = self._to_tensor(value)

        # also update the priority queue
        prio = self.priority_queue.get_max_priority() # const time
        self.priority_queue.new_experience(priority=prio, e_id=int(self.ptr)) # log time

        # update self.ptr and current size of the buffer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # occasionally completely sort the priority array
        self.sort_counter += 1
        if self.sort_counter == self.sort_interval:
            self.priority_queue.balance_tree()
            self.sort_counter = 0

    def sample(self, device='cpu'):

        if self.size < self.sample_start:
            sys.stderr.write('Record size less than learn start! Sample failed\n')
            return False, False, False

        # determine which buckets to use
        buckets_idx = max(1, np.floor(self.size / self.max_size * self.num_dist_parts)) 
        buckets = self.dists[buckets_idx]["buckets"]

        # stratified sampling...
        prio_indices = []
        for bucket_id in range(self.batch_size):
            if buckets[bucket_id] + 1 == buckets[bucket_id + 1]:
                index = buckets[bucket_id] + 1
            else:
                index = np.random.randint(buckets[bucket_id] + 1, buckets[bucket_id + 1])
            prio_indices.append(index)
        #retrieve buffer indices
        ind = np.array(self.priority_queue.priority_to_experience(prio_indices))

        if self.extra:
            extra = {key: value[ind].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        # store indices for updating the prios
        extra["buffer_id"] = ind

        # compute importance sampling weight
        pdf = self.dists[buckets_idx]['pdf']
        probs = np.array([pdf[rank-1] for rank in prio_indices])
        ws = np.power(probs * len(pdf), -self.beta)
        ws = ws / np.max(ws) # normalize
        extra["importance_weight"] = torch.from_numpy(ws).to(device) 

        #build the batch and return it
        batch = Batch(
            state = self.state[ind].to(device),
            action = self.action[ind].to(device), 
            next_state = self.next_state[ind].to(device), 
            reward = self.reward[ind].to(device), 
            not_done = self.not_done[ind].to(device), 
            extra = extra
        )
        return batch

    def get_all(self, device='cpu'):
        if self.extra:
            extra = {key: value[:self.size].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[:self.size].to(device),
            action = self.action[:self.size].to(device), 
            next_state = self.next_state[:self.size].to(device), 
            reward = self.reward[:self.size].to(device), 
            not_done = self.not_done[:self.size].to(device), 
            extra = extra
        )
        return batch

    def update_priorities(self, new_priorities, buffer_index):
        """
        update priority according indices and new priorities
        :param new_priorities: list of new priorities
        :transition_index: list of transition BUFFER indices (make sure that these are indices of the buffer...)
        :return: None
        """
        for prio, i in zip(new_priorities, buffer_index):
            self.priority_queue.update(prio, i)


class PrioritizedReplayBuffer2(object):
    def __init__(self, state_shape:tuple, action_dim: int, batch_size: int, max_size=int(1e6), alpha: float = 0.5, beta: float = 0.5, sort_interval: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.sort_counter = 0

        dtype = torch.uint8 if len(state_shape) == 3 else torch.float32 # unit8 is used to store images
        self.state = torch.zeros((max_size, *state_shape), dtype=dtype)
        self.action = torch.zeros((max_size, action_dim), dtype=dtype)
        self.next_state = torch.zeros((max_size, *state_shape), dtype=dtype)
        self.reward = torch.zeros((max_size, 1), dtype=dtype)
        self.not_done = torch.zeros((max_size, 1), dtype=dtype)
        self.extra = {}

        self.priority_arr = []
        self.alpha = alpha
        self.beta = beta
        self.sort_interval = sort_interval
        self.batch_size = batch_size
        self.bucket_ids = []

    def _to_tensor(self, data, dtype=torch.float32):   
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def add(self, state, action, next_state, reward, done, extra:dict=None):
        debug = False
        if debug:
            print("Adding transition to replay buffer...")
            print("Prev Qeue: ", self.priority_arr)
            print("Prev rewards: ", self.reward)

        if self.size < self.max_size:
            add_ptr = self.ptr
            self.ptr += 1
            self.size += 1
            heapq.heappush(self.priority_arr, (np.inf, add_ptr))
        else:
            add_ptr = self.priority_arr[0][1] # lowest prio experience
            heapq.heapreplace(self.priority_arr, (np.inf, add_ptr))
    	
        if debug: 
            print("Writing to position: ", add_ptr)

        self.state[add_ptr] = self._to_tensor(state, dtype=self.state.dtype)
        self.action[add_ptr] = self._to_tensor(action)
        self.next_state[add_ptr] = self._to_tensor(next_state, dtype=self.state.dtype)
        self.reward[add_ptr] = self._to_tensor(reward)
        self.not_done[add_ptr] = self._to_tensor(1. - done)

        if extra is not None:
            for key, value in extra.items():
                if key not in self.extra: # init buffer
                    self.extra[key] = torch.zeros((self.max_size, *value.shape), dtype=torch.float32)
                self.extra[key][add_ptr] = self._to_tensor(value)

        if debug:
            print("Result:")
            print("Updated Qeue: ", self.priority_arr)
            print("Updated rewards: ", self.reward)

        # occasionally completely sort the priority array
        self.sort_counter += 1
        if self.sort_counter == self.sort_interval:
            self.priority_arr = sorted(self.priority_arr)
            self.sort_counter = 0
            if debug:
                print("Sorted the queue:", self.priority_arr)

    def sample(self, device='cpu'):
        debug = False
        f = (lambda a: np.minimum(self.size-1, np.floor(self.size * a)))
        priority_arr_id = f(np.random.power(self.alpha + 1, size=self.batch_size)).astype(np.int32)
        ind = np.array([self.priority_arr[i][1] for i in priority_arr_id])

        den = functools.reduce(lambda a, b: a + (1 / b) **self.alpha, range(1, self.size+1), 0)
        props = ((priority_arr_id + 1)**self.alpha)
        importance_sampling_weight = (den / (props * self.size))**self.beta
        importance_sampling_weight /= np.max(importance_sampling_weight)

        if debug:
            print("Sampling from the priority queue...")
            print("Prio Queue: ", self.priority_arr)
            print("Sampled ids: ", priority_arr_id)
            print("Csp indices: ", ind)
            print("Importance Sampling Weights: ", importance_sampling_weight)

        if self.extra:
            extra = {key: value[ind].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        extra["prio_id"] = priority_arr_id
        extra["importance_weight"] = torch.from_numpy(importance_sampling_weight).to(device)  

        batch = Batch(
            state = self.state[ind].to(device),
            action = self.action[ind].to(device), 
            next_state = self.next_state[ind].to(device), 
            reward = self.reward[ind].to(device), 
            not_done = self.not_done[ind].to(device), 
            extra = extra
        )
        return batch

    def get_all(self, device='cpu'):
        if self.extra:
            extra = {key: value[:self.size].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[:self.size].to(device),
            action = self.action[:self.size].to(device), 
            next_state = self.next_state[:self.size].to(device), 
            reward = self.reward[:self.size].to(device), 
            not_done = self.not_done[:self.size].to(device), 
            extra = extra
        )
        return batch

    def update_priorities(self, new_priorities, transition_index):
        debug=False
        if debug:
            print("Updating priorities...")
            print("Prio Queue before: ", self.priority_arr)
            print("New Prios: ", new_priorities)
            print("Change ids: ", transition_index)
        for prio, i in zip(new_priorities, transition_index):
            j = self.priority_arr[i][1]
            self.priority_arr[i] = (prio, j)

        if debug: 
            print("Updated but not sorted: ", self.priority_arr)

        heapq.heapify(self.priority_arr)
        #self.priority_arr = sorted(self.priority_arr)
        if debug: 
            print("Final Prio Queue: ", self.priority_arr)


if __name__ == "__main__":
    max_size = 1000
    dist_parts = 100
    batch_size = 10
    alpha = 0.5

    res = {}
    dist_num = 1
    size_inc = int(np.floor(max_size / dist_parts))

    # for each buffer size compute the buckets
    for buffersize in range(size_inc, max_size+1, size_inc):
        ps = [np.power(float(x), -alpha) for x in range(1, buffersize + 1)]
        den = np.sum(ps)
        pdf = [x / den for x in ps]
        cdf = np.cumsum(pdf)
        buckets = [0] * (batch_size+1)
        buckets[batch_size] = buffersize # last bucket border is always the end of the buffer
        step = 1.0 / float(batch_size) # probability of each bucket region
        buffer_idx = 1
        for bucket_i in range(1, batch_size):
            while cdf[buffer_idx] < step:
                buffer_idx += 1
            buckets[bucket_i] = buffer_idx
            buffer_idx += 1
            step += 1.0 / float(batch_size)
        distribution = {'pdf': np.asarray(pdf),
                        'buckets': buckets}
        res[dist_num] = distribution
        dist_num+= 1
        # print(np.array(pdf))
        # print(cdf)
        print(buckets)
