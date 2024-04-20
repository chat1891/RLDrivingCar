import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils.envs
import utils.torch
import utils.seed
import utils.common
import utils.buffers
import RacingEnvironment
import json
import matplotlib.pyplot as plt
import warnings
import tqdm

warnings.filterwarnings("ignore")

class A2C:
    # Constants
    SEEDS = [1, 2]
    t = utils.torch.TorchHelper()
    DEVICE = t.device
    OBS_N = 17               # State space size, number of ray + velocity
    ACT_N = 5               # Action space size
    MINIBATCH_SIZE = 10     # How many examples to sample per train step
    GAMMA = 0.99            # Discount factor in episodic reward objective
    LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
    TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
    TRAIN_EPOCHS = 2       # Train for these many epochs every time
    EPISODES = 10000          # Total number of episodes to learn over
    TEST_EPISODES = 1       # Test episodes after every train episode
    HIDDEN = 512            # Hidden nodes
    STARTING_EPSILON = 1.0  # Starting epsilon
    STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
    EPSILON_END = 0.01      # At the end, keep epsilon at this value

    # Global variables
    EPSILON = STARTING_EPSILON
    
    def __init__(self, fname='a2c_file.h5'):
        pass

    def create_everything(self, seed):
        class ActorCritic(nn.Module):
            def __init__(self, input_dim, output_dim, hidden_dim=128):
                super(ActorCritic, self).__init__()
                self.actor = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    nn.Softmax(dim=-1)
                )
                self.critic = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )

            def forward(self, x):
                policy = self.actor(x)
                value = self.critic(x)
                return policy, value
        
        env = RacingEnvironment.RacingEnvironment()
        env.fps = 60

        test_env = RacingEnvironment.RacingEnvironment()
        test_env.fps = 60

        actor_critic = ActorCritic(self.OBS_N, self.ACT_N).to(self.DEVICE)
        optimizer = optim.Adam(actor_critic.parameters(), lr=self.LEARNING_RATE)

        self.actor_critic = actor_critic 

        return env, test_env, actor_critic, optimizer

    def policy(self, env, obs):
        if obs is None:
            return np.random.randint(self.ACT_N)  # Return a random action if the state is None

        obs = self.t.f(obs).view(-1, self.OBS_N).float().to(self.DEVICE)

        policy, _ = self.actor_critic(obs)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample().cpu().numpy()[0]

        return action


    def compute_returns(self, rewards, dones, next_value):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.GAMMA * R * (1 - dones[step])
            returns.insert(0, R)
        return returns

    def update_networks(self, optimizer, log_probs, values, rewards, dones, next_value):
        returns = self.compute_returns(rewards, dones, next_value)
        returns = torch.tensor(returns).to(self.DEVICE)
        
        log_probs = torch.stack(log_probs).to(self.DEVICE)
        values = torch.stack(values).to(self.DEVICE)
        
        advantage = returns - values

        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, seed=1):
        loss_history = []

        env, test_env, actor_critic, optimizer = self.create_everything(seed)

        testRs = []
        last25testRs = []

        print("Training:")
        pbar = tqdm.trange(self.EPISODES)

        for epi in pbar:
            states = []
            actions = []
            rewards = []
            dones = []
            log_probs = []
            values = []

            state = env.reset()
            done = False

            while not done:
                action = self.policy(env, state)
                next_state, reward, done = env.step(action)
                
                if next_state is None:
                    break

                env.render() 

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                state = next_state

                obs = self.t.f(state).view(-1, self.OBS_N).float().to(self.DEVICE)
                policy, value = actor_critic(obs)
                dist = torch.distributions.Categorical(policy)
                log_prob = dist.log_prob(torch.tensor([actions[-1]]).to(self.DEVICE))

                log_probs.append(log_prob)
                values.append(value)

            if next_state is None:
                continue

            next_value = 0 if done else actor_critic(self.t.f(next_state).view(-1, self.OBS_N).float().to(self.DEVICE))[1].cpu().data.numpy()[0]

            loss = self.update_networks(optimizer, log_probs, values, rewards, dones, next_value)
            loss_history.append(loss)

            Rews = []
            for epj in range(self.TEST_EPISODES):
                state = test_env.reset()
                done = False
                ep_reward = 0
                
                while not done:
                    obs = self.t.f(state).view(-1, self.OBS_N).float().to(self.DEVICE)
                    policy, _ = actor_critic(obs)
                    dist = torch.distributions.Categorical(policy)
                    action = dist.sample().cpu().numpy()[0]
                    
                    state, reward, done, _ = test_env.step(action)
                    ep_reward += reward
                
                Rews.append(ep_reward)
            
            testRs.append(sum(Rews)/self.TEST_EPISODES)
            last25testRs.append(sum(testRs[-25:])/len(testRs[-25:]))

            pbar.set_description("R25(%g)" % (last25testRs[-1]))
            

            env.render()

            if epi % 10 == 0 and epi > 10:
                self.save_model('a2c_model.pth')
                with open('last25testRs_a2c.json', 'w') as f:
                    json.dump(last25testRs, f)
                with open('loss_history_a2c.json', 'w') as f:
                    json.dump(loss_history, f)

        with open('last25testRs_a2c.json', 'w') as f:
            json.dump(last25testRs, f)
        with open('loss_history_a2c.json', 'w') as f:
            json.dump(loss_history, f)
        
        pbar.close()
        print("Training finished!")

        return last25testRs


    def plot_arrays(self, vars, color, label):
        mean = vars
        std = np.std(vars, axis=0)
        plt.plot(range(len(mean)), mean, color=color, label=label)
        plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std, 200), color=color, alpha=0.3)
        plt.savefig('a2c_loss.png')

    def loadLast25TestRs(self):
        with open('last25testRs_a2c.json', 'r') as f:
            last25testRs = json.load(f)
            return last25testRs

    def save_model(self, filepath='a2c_model.pth'):
        torch.save(self.actor_critic.state_dict(), filepath)
        print(f'Model saved to {filepath}')

    def load_model(self, filepath='a2c_model.pth'):
        checkpoint = torch.load(filepath, map_location=self.DEVICE)
        self.actor_critic.load_state_dict(torch.load(filepath, map_location=self.DEVICE))
        print(f'Model loaded from {filepath}')

