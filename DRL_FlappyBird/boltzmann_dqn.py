import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from game.flappy_bird import GameState


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data

def boltzman(x, beta, normalize = True):
    if normalize == True:
        b = torch.max(beta * x)
    else:
        b = 0
    
    expSum = torch.exp(beta * x - b)
    denom = expSum.sum()
    numer = x * expSum
    numer = numer.sum()

    return numer/denom


def train(model, start, temperature = 1,beginning_episode = 0, beginning_iteration = 0):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState()

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.initial_epsilon #self.initial_epsilon = 0.1
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    iteration_list = []
    reward_list = []
    Q_list = []
    
    iteration = beginning_iteration
    episode = beginning_episode
    tot_rew = []
    reward_list_episode = []
    game_run = [] #record episode numbers

    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32) #tensor([0., 0.])
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon #returns True/False
        if random_action:
            print("\n")
            print("Performed random action!")
            print("\n")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action #choose random action from [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        else torch.argmax(output)][0] #choose greedy action (as per exploration vs exploitation)

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        reward_list_episode.append(reward.numpy()[0][0])

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        if terminal == True:
            #record rewards for each episode
            episode_reward = sum(reward_list_episode)
            print("\n")
            print("EPISODE:", episode)
            print("REWARD FOR EPISODE:", episode_reward)
            print("\n")
            tot_rew.append(episode_reward)
            reward_list_episode = []
            game_run.append(episode)
            episode += 1 #update episode number

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size)) #ex: list1 = [1, 2, 3, 4, 5], sample(list1,3) -> [2, 3, 5]

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        output_1_batch = model(state_1_batch)
        output_1_batch_without_grad = output_1_batch.detach().float()

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * boltzman(output_1_batch_without_grad[i],temperature)
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()
        out_put_without_grad = output.detach().float()

        # calculate loss
        loss = criterion(q_value, y_batch) #https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial 
        print("LOSS:",loss.data)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        iteration_list.append(iteration)
        reward_list.append(reward.numpy()[0][0])
        Q_list.append(boltzman(out_put_without_grad,temperature).item())

        if iteration % args.save_every == 0:
            torch.save(model, args.model_save_path + "/boltzmann_model_" + str(iteration) + ".pth")
            np.savetxt(args.results_save_path + "/reward_loss_" + str(iteration) + ".csv", np.array(reward_list))
            np.savetxt(args.results_save_path + "/max_Q_" + str(iteration) + ".csv", np.array(Q_list))
            np.savetxt(args.results_save_path + "/Iteration_" + str(iteration) + ".csv", np.array(iteration_list))
            np.savetxt(args.game_run_path + "/total_reward" + str(iteration) + "_ep_" + str(episode) + ".csv", np.array(tot_rew))
            np.savetxt(args.game_run_path + "/game_run" + str(iteration) + "_ep_" + str(episode) + ".csv", np.array(game_run))


        print("episode:", episode, "iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Boltzman Q:",
              boltzman(out_put_without_grad,temperature).item())

def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')

    #load paths
    parser.add_argument("--model_save_path", type=str, 
                        help='path to save model weights')
    parser.add_argument("--load_path", type=str, 
                        help='path to model to be loaded')
    parser.add_argument("--results_save_path", type=str, 
                        help='path to save iteration results')
    parser.add_argument("--game_run_path", type=str, 
                        help='path to save episode results')

    #load run mode
    parser.add_argument("--mode", type=str, 
                        help='mode to run')

    #numerical params
    parser.add_argument("--save_every", type=int, default=50000, 
                        help='frequency to save model and statistics')
    
    parser.add_argument("--alpha", type=int, default=10, 
                        help='alpha value in soft_MM')

    parser.add_argument("--temperature", type=int, default=1000, 
                        help='deepmellow temperature')

    parser.add_argument("--beginning_episode", type=int, default=0, 
                        help='beginning_episode')

    parser.add_argument("--beginning_iteration", type=int, default=0, 
                        help='beginning_iteration')
            
    return parser


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            args.load_path,
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)

    elif mode == 'train':
        torch.manual_seed(222)  #fix random seed
        model = NeuralNetwork() #initialize model class

        if cuda_is_available:  # put on GPU if CUDA is available
            print("GPU model")
            model = model.cuda()

        #model.apply(init_weights)
        start = time.time()

        train(model, start,temperature = args.temperature, beginning_episode = args.beginning_episode, beginning_iteration = args.beginning_iteration)

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    torch.manual_seed(222)
    random.seed(222)
    main(args.mode)