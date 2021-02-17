import pickle
import matplotlib.pyplot as plt

rewards = pickle.load(open('./Reward.txt', 'rb'))

unit_len = 250
avg_rewards = []
for i in range(len(rewards)):
    if (i + unit_len) < len(rewards):
        if (i % unit_len) != i:
            avg_rewards.append(sum(rewards[i:i+unit_len])/unit_len)

# plt.plot(range(len(rewards)), rewards, 'b-')
plt.plot(range(len(avg_rewards)), avg_rewards, 'r-', linewidth='3')
plt.show()