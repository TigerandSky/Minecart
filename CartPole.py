import gym
import tflearn as tf
import numpy as np
env = gym.make('CartPole-v0')
score_req = 100
goal_steps = 1000
inital_games = 100000
LR = 1e-3
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
            score += reward
            if x > 0:
                game_memory.append([prev_observation, int(action)])
            prev_observation = observation
            if done:
                break
        if score > score_req:
            accepted_score.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                training_data.append([data[0], output])
    print(accepted_score)
    return training_data




# for  i in range(100):
#     observation = env.reset()
#     for t in range(200):
#         env.render()
#         #print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         # if done:
#         #     break
l = inital_population()
print(l)
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

def play_with_model(model):
   scores = []
   choices = []
   print("playing with the Trained Model.....")
   for each_game in range(10):
       score = 0
       game_memory = []
       prev_obs = []
       env.reset()
       for _ in range(goal_steps):
           env.render()
           if len(prev_obs) == 0:
               action = env.action_space.sample()
           else:
               action = np.argmax(model.predict([prev_obs])[0])
           choices.append(action)

           new_observation, rew, done, info = env.step(action)
           prev_obs = new_observation
           game_memory.append([new_observation, action])
           score += rew
           if done:
               break
           #if done and score != 200:
           #    break
       scores.append(score)

   print(scores)
model=train_model(inital_population())
play_with_model(model)

