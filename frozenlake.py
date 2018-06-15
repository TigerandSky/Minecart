#load the frameword
import gym, random
#load the frozenLake open environment
env = gym.make('FrozenLake-v0')
#Each environment has a 'obseravational space' and action space
env.reset()
print(env.observation_space)
print(env.action_space)#throuhg the trail
#0 = left
#1 = down
#2 = right
#3 = up
score = 0
env.reset()
for i in range(1000):
    env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        print("observation: " + str(obs))
        env.render()
        if done:
            break
    if rew == 1:
        score += 1
print('')
print('Out of 1000 times I won ' + str(score)+ ' times.')
print('I win 1.35% of the time on average')

