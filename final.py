import gym
import tflearn as tf
import numpy as np
env = gym.make("Pendulum-v0")
env.reset()
score_req = 0
goal_steps = 2000
inital_games = 200
LR = 1e-3
# for i in range(3):
#   observation = env.reset()
#   for _ in range(1000):
#     env.render()
#     action = env.action_space.sample() # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)

def inital_population():
    training_data = []
    accepted_score = []
    print("Playing random games")
    for i in range(inital_games):
        env.reset()
        game_memory = []
        prev_observation = []
        score = 0
        for x in range(goal_steps):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            action0 = 0  # do nothing
            score += reward
            #print(score)
            if x > 0:
                game_memory.append([prev_observation, int(action)])
            prev_observation = observation
            if done:
                break
        if score > inital_games:
            accepted_score.append(score)
            #print(game_memory)
            for data in game_memory:
                #print(training_data)
                if data[1] == 0:
                    output = [0,1]
                else:# data[1] == 0:
                    output = [1,0]
                training_data.append([data[0], output])
    print(accepted_score)
    return training_data
def neural_net_model(input_size):
   network = tf.input_data(shape=[None, input_size], name='input')

   network = tf.fully_connected(network,128,activation = 'relu')
   network = tf.dropout(network, 0.8)

   network = tf.fully_connected(network, 256, activation= 'relu', name= "hlayer1")
   network = tf.dropout(network, 0.8)

   network = tf.fully_connected(network, 512, activation= 'relu', name= "hlayer2")
   network = tf.dropout(network, 0.8)

   network = tf.fully_connected(network, 256, activation= 'relu', name= "hlayer3")
   network = tf.dropout(network, 0.8)

   network = tf.fully_connected(network, 128, activation= 'relu', name= "hlayer4")
   network = tf.dropout(network, 0.8)

   network= tf.fully_connected(network, 2, activation= 'softmax', name= "out")
   network = tf.regression(network, learning_rate= LR)

   model = tf.DNN(network,tensorboard_dir= 'log')

   return model
def train_model(training_data):
   x = [i[0] for i in training_data]
   y = []
   for i in training_data:
       y.append(i[1])
   model = neural_net_model(input_size=len(x[0]))
   model.fit(x, y, n_epoch=5, show_metric= True, run_id= 'openai_learning')

   return model
model=train_model(inital_population())