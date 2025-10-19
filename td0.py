import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use headless backend for faster plotting
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

# ==================== TD(0) AGENT ====================

class TD0Agent:
    def __init__(self, num_actions=2, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(num_actions))
        self.td_errors = []
        self.policy = defaultdict(lambda: 0)

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        """Update Q-values using TD(0) method"""
        # Calculate TD error
        current_q = self.Q[state][action]
        next_q = np.max(self.Q[next_state])
        td_error = reward + self.gamma * next_q - current_q
        
        # Update Q-value
        self.Q[state][action] += self.alpha * td_error
        self.td_errors.append(td_error)
        
        # Update policy
        self.policy[state] = np.argmax(self.Q[state])
        
        return td_error

def train_td0(num_episodes=100, max_steps=1000):
    """Train using TD(0) learning"""
    agent = TD0Agent()
    all_returns = []
    episode_lengths = []
    all_td_errors = []
    
    try:
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            traci.load(["-c", "rohit.sumocfg"])
            total_reward = 0
            steps = 0
            episode_td_errors = []
            
            # Run episode
            state = discretize_state(get_state())
            while traci.simulation.getMinExpectedNumber() > 0 and steps < max_steps:
                action = agent.select_action(state)
                
                # Take action and observe result
                apply_action(action)
                traci.simulationStep()
                
                next_state = discretize_state(get_state())
                reward = compute_reward()
                
                # Update agent
                td_error = agent.update(state, action, reward, next_state)
                episode_td_errors.append(td_error)
                
                total_reward += reward
                state = next_state
                steps += 1
            
            # Store metrics
            all_returns.append(total_reward)
            episode_lengths.append(steps)
            current_td_error = np.mean(episode_td_errors)
            all_td_errors.append(current_td_error)
            
            # Print immediate feedback
            print(f"\rEpisode {episode + 1}/{num_episodes} | "
                  f"Reward: {total_reward:.0f} | "
                  f"Steps: {steps} | "
                  f"TD Error: {current_td_error:.4f} | "
                  f"Avg Return: {np.mean(all_returns[-10:]):.0f}", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        traci.close()
    
    return agent, all_returns, episode_lengths, all_td_errors

def plot_results(returns, lengths, td_errors):
    """Plot training results"""
    plt.figure(figsize=(15, 5))
    
    # Plot returns
    plt.subplot(1, 3, 1)
    plt.plot(returns, 'b-', label='Returns')
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.legend()
    
    # Plot episode lengths
    plt.subplot(1, 3, 2)
    plt.plot(lengths, 'r-', label='Steps')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.grid(True)
    plt.legend()
    
    # Plot TD errors
    plt.subplot(1, 3, 3)
    plt.plot(td_errors, 'g-', label='TD Error')
    plt.xlabel('Episode')
    plt.ylabel('Average TD Error')
    plt.title('TD Error')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('td0_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'td0_results.png'")
    plt.show()

if __name__ == "__main__":
    print("Starting TD(0) training...")
    
    # Initialize SUMO
    sumo_binary = "sumo-gui"
    sumo_cmd = [sumo_binary, "-c", "rohit.sumocfg"]
    
    try:
        traci.start(sumo_cmd)
        print("SUMO initialized successfully")
        
        # Train agent
        agent, returns, lengths, td_errors = train_td0(num_episodes=10)
        
        # Plot results
        plot_results(returns, lengths, td_errors)
        
        print("\n====== Training Results ======")
        print(f"Episodes completed: {len(returns)}")
        print(f"Final average return: {np.mean(returns[-10:]):.2f}")
        print(f"Best return: {max(returns):.2f}")
        print(f"Average episode length: {np.mean(lengths):.1f} steps")
        print(f"Final TD error: {td_errors[-1]:.4f}")
        print("==============================")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            traci.close()
        except:
            pass