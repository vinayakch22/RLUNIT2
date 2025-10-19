import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use headless backend
import matplotlib.pyplot as plt
import traci
import os
import sys
from collections import defaultdict

# Set up SUMO environment
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = 'C:\\Program Files (x86)\\Eclipse\\Sumo'

# ==================== STATE AND ACTION UTILITIES ====================

def discretize_state(state, bins=5):
    """Discretize continuous state into bins for tabular methods"""
    discretized = []
    for val in state:
        if val < 0:
            val = 0
        bin_idx = min(int(val / 10), bins - 1)
        discretized.append(bin_idx)
    return tuple(discretized)

def get_state():
    """Get current state from SUMO"""
    try:
        # Get waiting times for all lanes
        waiting_times = [traci.lane.getWaitingTime(lane_id) for lane_id in traci.lane.getIDList()]
        # Get vehicle counts
        vehicle_counts = [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in traci.lane.getIDList()]
        # Get traffic light phase
        try:
            current_phase = traci.trafficlight.getPhase("B2")
        except:
            current_phase = 0
            
        return np.array(waiting_times + vehicle_counts + [current_phase])
    except Exception as e:
        print(f"Error in get_state: {e}")
        return np.zeros(1)

def compute_reward():
    """Compute reward based on waiting times and vehicles cleared"""
    try:
        # Penalize waiting time
        total_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in traci.lane.getIDList())
        
        # Reward for vehicles that have cleared the intersection
        vehicles_cleared = sum(traci.edge.getLastStepVehicleNumber(edge_id) for edge_id in traci.edge.getIDList())
        
        return vehicles_cleared - total_waiting_time
    except Exception as e:
        print(f"Error in compute_reward: {e}")
        return 0

def apply_action(action):
    """Apply action to traffic light"""
    try:
        if action == 0:
            traci.trafficlight.setPhase("B2", 0)  # Green for EW
        elif action == 1:
            traci.trafficlight.setPhase("B2", 2)  # Green for NS
    except Exception as e:
        print(f"Error in apply_action: {e}")

# ==================== MONTE CARLO AGENT ====================

class MonteCarloAgent:
    def __init__(self, num_actions=2, gamma=0.99, epsilon=0.1):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(num_actions))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.policy = defaultdict(lambda: 0)

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.Q[state])

    def update(self, episode):
        """Update Q-values using Monte Carlo first-visit method"""
        states, actions, rewards = zip(*episode)
        
        # Calculate returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # Update Q-values and policy
        seen = set()
        for t in range(len(episode)):
            state = states[t]
            action = actions[t]
            if (state, action) not in seen:
                self.returns[state][action].append(returns[t])
                self.Q[state][action] = np.mean(self.returns[state][action])
                self.policy[state] = np.argmax(self.Q[state])
                seen.add((state, action))

def train_monte_carlo(num_episodes=100, max_steps=1000):
    """Train using Monte Carlo control"""
    agent = MonteCarloAgent()
    all_returns = []
    episode_lengths = []
    
    def update_plot():
        """Update the learning curves plot"""
        plt.clf()  # Clear the current figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot returns
        ax1.plot(all_returns, 'b-', label='Returns')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Return')
        ax1.set_title('Learning Curve')
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(episode_lengths, 'r-', label='Length')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('monte_carlo_results.png')
        plt.close()  # Close the figure to free memory
    
    try:
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            traci.load(["-c", "rohit.sumocfg"])
            episode_data = []
            total_reward = 0
            steps = 0
            
            # Run episode
            state = discretize_state(get_state())
            while traci.simulation.getMinExpectedNumber() > 0 and steps < max_steps:
                action = agent.select_action(state)
                
                # Take action and observe result
                apply_action(action)
                traci.simulationStep()
                
                next_state = discretize_state(get_state())
                reward = compute_reward()
                
                episode_data.append((state, action, reward))
                total_reward += reward
                state = next_state
                steps += 1
            
            # Update agent
            agent.update(episode_data)
            
            # Store metrics and update plot
            all_returns.append(total_reward)
            episode_lengths.append(steps)
            update_plot()  # Update the plot after each episode
            
            # Print immediate feedback
            print(f"\rEpisode {episode + 1}/{num_episodes} | "
                  f"Reward: {total_reward:.0f} | "
                  f"Steps: {steps} | "
                  f"Avg Return: {np.mean(all_returns[-10:]):.0f} | "
                  f"Best: {max(all_returns):.0f}", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        traci.close()
    
    return agent, all_returns, episode_lengths

def plot_results(returns, lengths):
    """Plot training results"""
    # Switch to a non-interactive backend for faster plotting
    plt.switch_backend('Agg')
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot returns
    ax1.plot(returns, 'b-', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Return')
    ax1.set_title('Learning Curve')
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax2.plot(lengths, 'r-', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png', dpi=300, bbox_inches='tight')
    
    # Switch back to TkAgg for display
    plt.switch_backend('TkAgg')
    plt.show()

if __name__ == "__main__":
    print("Starting Monte Carlo training...")
    
    # Initialize SUMO
    sumo_binary = "sumo-gui"
    sumo_cmd = [sumo_binary, "-c", "rohit.sumocfg"]
    
    try:
        traci.start(sumo_cmd)
        print("SUMO initialized successfully")
        
        # Train agent
        agent, returns, lengths = train_monte_carlo(num_episodes=10)
        
        print("\nTraining completed!")
        print(f"Final average return: {np.mean(returns[-10:]):.2f}")
        print("\nPlot has been saved as 'monte_carlo_results.png'")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            traci.close()
        except:
            pass