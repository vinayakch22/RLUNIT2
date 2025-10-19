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

# TD(0) implementation
def td_zero(num_episodes=100, alpha=0.1, gamma=0.99):
    Q = {}  # State-action value function
    actions = [0, 1]  # 0: EW green, 1: NS green

    total_rewards = []  # Track total reward per episode
    avg_td_errors = []  # Track average TD error per episode

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        traci.load(["-c", "C:\\Users\\rnutr\\Videos\\RL_New\\rayudu.sumocfg"])  # Reload SUMO
        state = get_state()
        total_reward = 0
        td_errors = []  # Track TD errors within the episode

        while traci.simulation.getMinExpectedNumber() > 0:
            # Choose an action
            action = random.choice(actions)
            take_action(action)
            traci.simulationStep()

            # Observe next state and reward
            next_state = get_state()
            reward = compute_reward()
            total_reward += reward  # Accumulate total reward

            # Initialize Q-values if state-action pairs are unseen
            if (state, action) not in Q:
                Q[(state, action)] = 0
            if (next_state, 0) not in Q:
                Q[(next_state, 0)] = 0
            if (next_state, 1) not in Q:
                Q[(next_state, 1)] = 0

            # TD(0) update
            max_q_next = max(Q[(next_state, a)] for a in actions)
            td_error = reward + gamma * max_q_next - Q[(state, action)]
            Q[(state, action)] += alpha * td_error
            td_errors.append(abs(td_error))  # Track absolute TD error

            # Move to the next state
            state = next_state

        # Track metrics
        total_rewards.append(total_reward)
        avg_td_errors.append(np.mean(td_errors))

    return Q, total_rewards, avg_td_errors

# Plotting the results
def plot_td_zero_results(total_rewards, avg_td_errors):
    plt.figure(figsize=(10, 6))

    # Total Reward vs Episodes
    plt.subplot(2, 1, 1)
    plt.plot(total_rewards, label="Total Reward", color="green")
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total Reward", fontsize=14)
    plt.title("Total Reward vs Episodes", fontsize=16)
    plt.grid(True)

    # Average TD Error vs Episodes
    plt.subplot(2, 1, 2)
    plt.plot(avg_td_errors, label="Average TD Error", color="blue")
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Average TD Error", fontsize=14)
    plt.title("Average TD Error vs Episodes", fontsize=16)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Run Simulation
if __name__ == "__main__":
    sumo_cfg_file = "C:\\Users\\rnutr\\Videos\\RL_New\\rayudu.sumocfg"
    initialize_sumo(sumo_cfg_file)
    Q, total_rewards, avg_td_errors = td_zero(num_episodes=100)
    traci.close()

    # Plot the results
    plot_td_zero_results(total_rewards, avg_td_errors)

    print("Learned Q-values:")
    for key, value in Q.items():
        print(f"State-Action: {key}, Value: {value}")
