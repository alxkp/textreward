import re
import numpy as np
import plotly.graph_objects as go

# Initialize empty lists to store the parsed data
episodes = []
dqn_losses = []
in_dist_rewards = []
out_dist_rewards = []

# Open the file and read its contents
with open('dqn_episodes.txt', 'r') as file:
    for line in file:
        # Use regular expressions to extract the desired information
        match = re.search(r'Episode: (\d+) \| time spent: [\d:.]+ \| dqn loss: ([\d.]+) \| overall rewards: ([\d.]+)/([\d.]+)', line)
        
        if match:
            episode = int(match.group(1))
            dqn_loss = float(match.group(2))
            in_dist_reward = float(match.group(3))
            out_dist_reward = float(match.group(4))
            
            episodes.append(episode)
            dqn_losses.append(dqn_loss)
            in_dist_rewards.append(in_dist_reward)
            out_dist_rewards.append(out_dist_reward)

# Print the parsed data
print("Episodes:", episodes)
print("DQN Losses:", dqn_losses)
print("In-Distribution Rewards:", in_dist_rewards)
print("Out-of-Distribution Rewards:", out_dist_rewards)


# Calculate polynomial fits
poly_degree = 2  # Degree of the polynomial fit

in_dist_fit = np.polyfit(episodes, in_dist_rewards, poly_degree)
in_dist_fit_fn = np.poly1d(in_dist_fit)
in_dist_fit_line = in_dist_fit_fn(episodes)

out_dist_fit = np.polyfit(episodes, out_dist_rewards, poly_degree)
out_dist_fit_fn = np.poly1d(out_dist_fit)
out_dist_fit_line = out_dist_fit_fn(episodes)

# Generate plotly figure and save it with title using graph objects
layout = go.Layout(title='DQN Losses Over Time', xaxis=dict(title='Episodes'), yaxis=dict(title='DQN Loss'))
trace = go.Scatter(x=episodes, y=dqn_losses, name='DQN Loss', mode='lines+markers')
fig = go.Figure(data=[trace], layout=layout)
fig.show()

trace3 = go.Scatter(
    x=episodes,
    y=in_dist_fit_line,
    name='In-Distribution Fit',
    mode='lines',
    line=dict(color='red', dash='dash')
)




# second figure
layout = go.Layout(title='In-Distribution Rewards Over Time', xaxis=dict(title='Episodes'), yaxis=dict(title='Rewards'))
trace1 = go.Scatter(x=episodes, y=in_dist_rewards, name='In-Distribution Reward', mode='lines+markers')

fig = go.Figure(data=[trace1, trace3], layout=layout)
fig.show()

# third figure
layout = go.Layout(title='Out-of-Distribution Rewards Over Time', xaxis=dict(title='Episodes'), yaxis=dict(title='Rewards'))
trace2 = go.Scatter(x=episodes, y=out_dist_rewards, name='Out-of-Distribution Reward', mode='lines+markers')
trace4 = go.Scatter(
    x=episodes,
    y=out_dist_fit_line,
    name='Out-of-Distribution Fit',
    mode='lines',
    line=dict(color='red', dash='dash')
)
fig = go.Figure(data=[trace2, trace4], layout=layout)
fig.show()
