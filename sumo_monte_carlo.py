import matplotlib.pyplot as plt
import traci
import numpy as np
import random

# Initialize SUMO
def initialize_sumo(sumo_cfg_file):
    sumo_binary = "sumo-gui"
    traci.start([sumo_binary, "-c", sumo_cfg_file])

# Define states, actions, and rewards
def get_state():
    state = []
    for lane_id in traci.lane.getIDList():
        state.append(traci.lane.getLastStepVehicleNumber(lane_id))
    return tuple(state)

def take_action(action):
    if action == 0:
        traci.trafficlight.setPhase("B2", 0)  # Green for EW
    elif action == 1:
        traci.trafficlight.setPhase("B2", 2)  # Green for NS

def compute_reward():
    vehicles_cleared = sum(traci.edge.getLastStepVehicleNumber(edge_id) for edge_id in traci.edge.getIDList())
    total_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in traci.lane.getIDList())
    return vehicles_cleared - total_waiting_time

# Monte Carlo simulation
def monte_carlo(num_episodes=100, gamma=0.99):
    Q = {}  # State-action value function
    returns = {}  # To track returns for each state-action pair
    actions = [0, 1]  # 0: EW green, 1: NS green

    total_rewards = []  # Track total reward per episode
    avg_returns = []  # Track average return per episode

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        traci.load(["-c", "C:\\Users\\rnutr\\Videos\\RL_New\\rayudu.sumocfg"])  # Reload SUMO
        state = get_state()
        episode_data = []
        total_reward = 0

        # Generate an episode
        while traci.simulation.getMinExpectedNumber() > 0:
            action = random.choice(actions)
            take_action(action)
            traci.simulationStep()
            
            next_state = get_state()
            reward = compute_reward()
            total_reward += reward
            
            episode_data.append((state, action, reward))
            state = next_state

        total_rewards.append(total_reward)  # Store total reward for this episode

        # Process episode data
        G = 0
        visited = set()
        returns_this_episode = []  # Store returns for this episode
        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward
            returns_this_episode.append(G)
            if (state, action) not in visited:
                if (state, action) not in returns:
                    returns[(state, action)] = []
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])
                visited.add((state, action))
        
        avg_returns.append(np.mean(returns_this_episode))  # Average return for the episode

    return Q, total_rewards, avg_returns

# Plotting the results
def plot_metrics(total_rewards, avg_returns):
    plt.figure(figsize=(12, 6))

    # Total Reward vs. Episodes
    plt.subplot(2, 1, 1)
    plt.plot(total_rewards, label="Total Reward", color="blue")
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total Reward", fontsize=14)
    plt.title("Total Reward vs. Episodes", fontsize=16)
    plt.grid(True)

    # Average Return (G) vs. Episodes
    plt.subplot(2, 1, 2)
    plt.plot(avg_returns, label="Average Return (G)", color="green")
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Average Return (G)", fontsize=14)
    plt.title("Average Return (G) vs. Episodes", fontsize=16)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Run Simulation
if __name__ == "__main__":
    sumo_cfg_file = "C:\\Users\\rnutr\\Videos\\RL_New\\rayudu.sumocfg"
    initialize_sumo(sumo_cfg_file)
    Q, total_rewards, avg_returns = monte_carlo(num_episodes=100)
    traci.close()

    # Plot the results
    plot_metrics(total_rewards, avg_returns)

    print("Learned Q-values:")
    for key, value in Q.items():
        print(f"State-Action: {key}, Value: {value}")
