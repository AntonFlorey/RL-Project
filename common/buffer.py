xfrom collections import namedtuple
import numpy as np
import torch
import heapq
import functools

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
        self.use_prios = self.compute_buckets()

        # pre compute buckets
    
    def compute_buckets(self):
        # compute partial sum of 1/p 
        den = functools.reduce(lambda a, b: a + (1 / b) **self.alpha, range(1, self.size+1), 0)
        # den = np.sum(np.array([1 / (i+1) ** self.alpha for i in range(1, self.size+1)]))

        self.bucket_ids = [-1]
        # loop variable        
        idx = 0
        # current probability mass count
        prob = 0
        while idx < self.size:
            # add current discrete probability mass to counter
            prob += ((1 / (idx + 1)) ** self.alpha) / den
            # if probability mass is greater than 1 / batch_size ...
            if prob >= 1 / self.batch_size:
                # ... add the index to the bucket_ids list
                self.bucket_ids.append(idx)
                # reset probability mass counter
                prob -= 1 / self.batch_size
            # increment index
            idx += 1
        # add the last index
        if len(self.bucket_ids) < self.batch_size + 1:
            self.bucket_ids.append(self.size - 1)

        # check if we have enough buckets
        if (len(self.bucket_ids) != self.batch_size + 1):
            return False

        return True

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
        
        # priority stuff TODO: drop the element with low priority 

        # priority_arr is a list of tuples (priority, index)
        # create a new tuple with -infty priority and the current index
        new_prio_elem = (-np.inf, self.ptr)
        # if the buffer is full ...
        if self.size == self.max_size:
            # ... replace the oldest element ([old_priority, self.ptr]) with the new one
            for i in range(self.max_size):
                if self.priority_arr[i][1] == self.ptr:
                    # replace the element
                    self.priority_arr[i] = new_prio_elem
                    # self.priority_arr is not a heap, so heapify
                    heapq.heapify(self.priority_arr)
                    break
        # if there is space in the buffer ...
        else:
            # ... add the new element to the list
            heapq.heappush(self.priority_arr, new_prio_elem)

        # update self.ptr and current size of the buffer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # occasionally completely sort the priority array
        # increment the sort counter
        self.sort_counter += 1
        # if the sort counter is equal to the sort interval
        if self.sort_counter == self.sort_interval:
            # ... sort the priority array
            self.priority_arr = sorted(self.priority_arr)
            # reset the sort counter
            self.sort_counter = 0

    def sample(self, device='cpu'):
        if self.bucket_ids[-1] != self.max_size - 1:
            self.use_prios = self.compute_buckets()
        priority_arr_id = []
        if not self.use_prios:
            priority_arr_id = np.random.randint(0, self.size, size=self.batch_size).tolist()
            ind = np.array([self.priority_arr[i][1] for i in priority_arr_id])
            importance_sampling_weight = np.ones(self.batch_size)
        else:
            ind = []
            props = []
            den = functools.reduce(lambda a, b: a + (1 / b) **self.alpha, range(1, self.size+1), 0)
            for sample in range(self.batch_size):
                if self.bucket_ids[sample + 1] == self.bucket_ids[sample] + 1:
                    i = self.bucket_ids[sample + 1]
                else:
                    i = np.random.randint(self.bucket_ids[sample]+1, self.bucket_ids[sample+1])

                props.append((i+1)**self.alpha)
                ind.append(self.priority_arr[i][1])
                priority_arr_id.append(i)
            ind = np.array(ind)
            importance_sampling_weight = (den / (np.array(props) * self.size))**self.beta
            importance_sampling_weight /= np.max(importance_sampling_weight)

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
        for prio, i in zip(new_priorities, transition_index):
            j = self.priority_arr[i][1]
            self.priority_arr[i] = (prio, j)
        heapq.heapify(self.priority_arr)


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

        #heapq.heapify(self.priority_arr)
        self.priority_arr = sorted(self.priority_arr)
        if debug: 
            print("Final Prio Queue: ", self.priority_arr)