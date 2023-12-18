from collections import deque
import random
import sys
import numpy as np
import torch
from model import DeepQN

#These are the exact same parameters and settings from PyBoy-RL's kirby
class LearningParameters():
    def __init__(self):
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999975 
        self.exploration_rate_min = 0.01

        self.deque_size = 500000
        self.batch_size = 64
        self.save_every = 1e4

        self.gamma = 0.8
        self.learning_rate = 0.0002
        self.learning_rate_decay = 0.9999985
        self.burnin = 1000  
        self.learn_every = 3  
        self.sync_every = 100

#Based on pytorch RL tutorial by yfeng997 used on Mario and Kirby's Dreamland 1 in PyBoy-RL
class AIKirby:

    def __init__(self, state_dim, action_space_dim, save_dir, date):
        self.params = LearningParameters()
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
        self.save_dir = save_dir
        self.date = date
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        
        self.net = DeepQN(self.state_dim, self.action_space_dim).to(device=self.device)

        self.exploration_rate = self.params.exploration_rate 
        self.exploration_rate_decay = self.params.exploration_rate_decay
        self.exploration_rate_min = self.params.exploration_rate_min
        self.curr_step = 0

        self.memory = deque(maxlen=self.params.deque_size)
        self.batch_size = self.params.batch_size
        self.save_every = self.params.save_every

        self.gamma = self.params.gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.params.learning_rate_decay)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = self.params.burnin
        self.learn_every = self.params.learn_every 
        self.sync_every = self.params.sync_every

    def act(self, state):
        # Decide between exploration and exploitation
        if random.random() < self.exploration_rate:
            # Exploration: choose a random action
            action_idx = random.randint(0, self.action_space_dim - 1)
        else:
            # Exploitation: choose the best action based on current knowledge

            # Convert state to tensor and process for neural network input
            state_tensor = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)

            # Obtain the output from the neural network
            neural_net_output = self.net(state_tensor, model="online")

            # Select the action with the highest value
            action_idx = torch.argmax(neural_net_output, axis=1).item()

        # Decrease exploration rate according to the decay factor
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # Increment the step counter
        self.curr_step += 1

        return action_idx


    def cache(self, state, next_state, action, reward, done):
        # Convert states, action, reward, and done flag to numpy arrays
        state = np.array(state)
        next_state = np.array(next_state)

        # Convert to PyTorch tensors and move to the specified device
        state_tensor = torch.tensor(state).float().to(self.device)
        next_state_tensor = torch.tensor(next_state).float().to(self.device)
        action_tensor = torch.tensor([action]).to(self.device)
        reward_tensor = torch.tensor([reward]).to(self.device)
        done_tensor = torch.tensor([done]).to(self.device)

        # Append the experience to the memory buffer
        self.memory.append((state_tensor, next_state_tensor, action_tensor, reward_tensor, done_tensor))


    def recall(self):
        # Randomly sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)

        # Separate the batch into individual components
        # torch.stack is used to concatenate a sequence of tensors along a new dimension
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        # Squeeze the tensors to remove any extra dimensions (of size 1)
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def learn(self):
        # Synchronize the Q-target network with the Q-online network periodically
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Save the model periodically
        if self.curr_step % self.save_every == 0:
            self.save()

        # Wait until enough experiences are collected before starting training
        if self.curr_step < self.burnin:
            return None, None

        # Learn only at specified intervals
        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample a batch of experiences from memory
        state, next_state, action, reward, done = self.recall()

        # Get the current Q-value estimates for the sampled states and actions
        td_est = self.td_estimate(state, action)

        # Calculate the target Q-values for the sampled next states
        td_tgt = self.td_target(reward, next_state, done)

        # Update the Q-online network by backpropagating the loss
        loss = self.update_Q_online(td_est, td_tgt)

        # Return the mean TD estimate and the loss for logging purposes
        return (td_est.mean().item(), loss)


    def update_Q_online(self, td_estimate, td_target):
        # Calculate the loss between the TD estimate and TD target
        loss = self.loss_fn(td_estimate, td_target)

        # Zero the gradients to prepare for a new gradient calculation
        self.optimizer.zero_grad()

        # Backpropagate the loss through the network
        loss.backward()

        # Perform a single optimization step to update the network weights
        self.optimizer.step()

        # Update the learning rate using the scheduler
        self.scheduler.step()

        # Return the loss value for logging or debugging
        return loss.item()


    def sync_Q_target(self):
        # Copy the weights from the online network to the target network
        self.net.target.load_state_dict(self.net.online.state_dict())


    def td_estimate(self, state, action):
        """
            Output is batch_size number of rewards = Q_online(s,a) * 32
        """
        modelOutPut = self.net(state, model="online")
        current_Q = modelOutPut[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # Get Q-value estimates for the next states using the online model
        next_state_Q = self.net(next_state, model="online")

        # Find the best action for each next state according to the online model
        best_action = torch.argmax(next_state_Q, axis=1)

        # Get Q-values for these best actions from the target model
        # This combines the selection of actions using the online network
        # with the evaluation of these actions using the target network
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]

        # Compute the TD target: reward + gamma * Q-value of the next state,
        # only updating the non-terminal states (where done is False)
        td_target = (reward + (1 - done.float()) * self.gamma * next_Q).float()

        return td_target

    def loadModel(self, path):
        dt = torch.load(path, map_location=torch.device(self.device))
        self.net.load_state_dict(dt["model"])
        self.exploration_rate = dt["exploration_rate"]
        print(f"Loading model at {path} with exploration rate {self.exploration_rate}")

    def saveHyperParameters(self):
        save_HyperParameters = self.save_dir / "hyperparameters"
        with open(save_HyperParameters, "w") as f:
            f.write(f"exploration_rate = {self.params.exploration_rate}\n")
            f.write(f"exploration_rate_decay = {self.params.exploration_rate_decay}\n")
            f.write(f"exploration_rate_min = {self.params.exploration_rate_min}\n")
            f.write(f"deque_size = {self.params.deque_size}\n")
            f.write(f"batch_size = {self.params.batch_size}\n")
            f.write(f"gamma (discount parameter) = {self.params.gamma}\n")
            f.write(f"learning_rate = {self.params.learning_rate}\n")
            f.write(f"learning_rate_decay = {self.params.learning_rate_decay}\n")
            f.write(f"burnin = {self.params.burnin}\n")
            f.write(f"learn_every = {self.params.learn_every}\n")
            f.write(f"sync_every = {self.params.sync_every}")

    def save(self):
        """
            Save the state to directory
        """
        save_path = (self.save_dir / f"kirby_net_0{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"KirbyNet saved to {save_path} at step {self.curr_step}")