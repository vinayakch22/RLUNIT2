import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import traci
import os
import sys
from collections import deque

# Set up SUMO environment
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = 'C:\\Program Files (x86)\\Eclipse\\Sumo'

# ==================== NEURAL NETWORK MODELS ====================

def build_value_function_network(input_dim, output_dim, learning_rate=0.001):
    """Build neural network for value function approximation"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='linear')  # Q-values for each action
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

def build_state_value_network(input_dim, learning_rate=0.001):
    """Build network for state value function V(s)"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # State value
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

# ==================== STATE AND REWARD FUNCTIONS ====================

def get_state():
    """Get comprehensive state representation from SUMO"""
    try:
        lane_ids = traci.lane.getIDList()
        
        # Traffic metrics per lane
        waiting_times = [traci.lane.getWaitingTime(lid) for lid in lane_ids]
        vehicle_counts = [traci.lane.getLastStepVehicleNumber(lid) for lid in lane_ids]
        speeds = [traci.lane.getLastStepMeanSpeed(lid) for lid in lane_ids]
        halting = [traci.lane.getLastStepHaltingNumber(lid) for lid in lane_ids]
        occupancy = [traci.lane.getLastStepOccupancy(lid) for lid in lane_ids]
        
        # Traffic light state
        try:
            current_phase = traci.trafficlight.getPhase("B2")
            phase_duration = traci.trafficlight.getPhaseDuration("B2")
            time_since_change = traci.trafficlight.getNextSwitch("B2") - traci.simulation.getTime()
        except:
            current_phase = 0
            phase_duration = 0
            time_since_change = 0
        
        # Combine all features
        state = np.array(waiting_times + vehicle_counts + speeds + halting + 
                        occupancy + [current_phase, phase_duration, time_since_change])
        
        # Normalize state (important for neural networks)
        state = state / (np.linalg.norm(state) + 1e-8)
        
        return state
    except Exception as e:
        print(f"Error in get_state: {e}")
        return np.zeros(10)

def compute_reward():
    """Multi-objective reward function"""
    try:
        lane_ids = traci.lane.getIDList()
        
        # Get metrics
        total_waiting = sum(traci.lane.getWaitingTime(lid) for lid in lane_ids)
        total_vehicles = sum(traci.lane.getLastStepVehicleNumber(lid) for lid in lane_ids)
        total_halting = sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lane_ids)
        avg_speed = np.mean([traci.lane.getLastStepMeanSpeed(lid) for lid in lane_ids])
        
        # Weighted reward
        reward = (
            (total_vehicles * 10.0) +      # Encourage throughput
            (avg_speed * 3.0) -             # Encourage higher speeds
            (total_waiting * 0.2) -         # Penalize waiting
            (total_halting * 8.0)           # Penalize stopped vehicles
        )
        
        return reward
    except Exception as e:
        return 0

def apply_action(action):
    """Apply action to traffic light"""
    try:
        phase_map = {0: 0, 1: 2}  # 0: NS-Green, 1: EW-Green
        traci.trafficlight.setPhase("B2", phase_map.get(action, 0))
    except Exception as e:
        print(f"Error in apply_action: {e}")

# ==================== TD(0) WITH FUNCTION APPROXIMATION ====================

def train_td0_function_approximation(num_episodes=100, gamma=0.99, epsilon=0.1, 
                                     epsilon_decay=0.995, epsilon_min=0.01,
                                     max_steps=1500):
    """Train using TD(0) with neural network function approximation"""
    
    # Initialize
    traci.start(["sumo", "-c", "rohit.sumocfg", "--start", "--quit-on-end"])
    state = get_state()
    input_dim = len(state)
    num_actions = 2
    traci.close()
    
    # Build Q-network (action-value function)
    q_network = build_value_function_network(input_dim, num_actions, learning_rate=0.001)
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    avg_td_errors = []
    avg_waiting_times = []
    avg_queue_lengths = []
    avg_speeds = []
    training_losses = []
    epsilon_values = []
    q_value_means = []
    q_value_stds = []
    action_distributions = []
    
    print("="*70)
    print("TD(0) WITH FUNCTION APPROXIMATION - TRAINING")
    print("="*70)
    print(f"State dimension: {input_dim}")
    print(f"Number of actions: {num_actions}")
    print(f"Episodes: {num_episodes}")
    print(f"Gamma (discount): {gamma}")
    print(f"Initial epsilon: {epsilon}")
    print("="*70)
    
    for episode in range(num_episodes):
        try:
            # Start SUMO
            traci.start(["sumo", "-c", "rohit.sumocfg", "--start", "--quit-on-end"])
            
            state = get_state()
            episode_reward = 0
            episode_td_errors = []
            episode_waiting = []
            episode_queues = []
            episode_speeds = []
            episode_losses = []
            episode_q_values = []
            action_counts = {0: 0, 1: 0}
            steps = 0
            
            while traci.simulation.getMinExpectedNumber() > 0 and steps < max_steps:
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(num_actions)
                else:
                    q_values = q_network.predict(state.reshape(1, -1), verbose=0)[0]
                    action = np.argmax(q_values)
                    episode_q_values.append(q_values)
                
                action_counts[action] += 1
                
                # Apply action and step simulation
                apply_action(action)
                for _ in range(5):  # Multiple steps per action
                    traci.simulationStep()
                    steps += 1
                
                # Observe next state and reward
                next_state = get_state()
                reward = compute_reward()
                episode_reward += reward
                
                # Get current Q-value predictions
                q_values_current = q_network.predict(state.reshape(1, -1), verbose=0)[0]
                q_values_next = q_network.predict(next_state.reshape(1, -1), verbose=0)[0]
                
                # TD(0) target: r + γ * max_a' Q(s', a')
                td_target = reward + gamma * np.max(q_values_next)
                td_error = td_target - q_values_current[action]
                episode_td_errors.append(abs(td_error))
                
                # Update Q-value for taken action
                target_q = q_values_current.copy()
                target_q[action] = td_target
                
                # Train network
                history = q_network.fit(
                    state.reshape(1, -1),
                    target_q.reshape(1, -1),
                    verbose=0,
                    epochs=1
                )
                episode_losses.append(history.history['loss'][0])
                
                # Track traffic metrics
                lane_ids = traci.lane.getIDList()
                episode_waiting.append(sum(traci.lane.getWaitingTime(lid) for lid in lane_ids))
                episode_queues.append(sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lane_ids))
                episode_speeds.append(np.mean([traci.lane.getLastStepMeanSpeed(lid) for lid in lane_ids]))
                
                state = next_state
            
            traci.close()
            
            # Decay epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            avg_td_errors.append(np.mean(episode_td_errors))
            avg_waiting_times.append(np.mean(episode_waiting))
            avg_queue_lengths.append(np.mean(episode_queues))
            avg_speeds.append(np.mean(episode_speeds))
            training_losses.append(np.mean(episode_losses))
            epsilon_values.append(epsilon)
            action_distributions.append(action_counts)
            
            # Q-value statistics
            if episode_q_values:
                q_vals_array = np.array(episode_q_values)
                q_value_means.append(np.mean(q_vals_array))
                q_value_stds.append(np.std(q_vals_array))
            else:
                q_value_means.append(0)
                q_value_stds.append(0)
            
            # Print progress
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f} | Steps: {steps}")
            print(f"  TD Error: {avg_td_errors[-1]:.4f} | Loss: {training_losses[-1]:.4f}")
            print(f"  Epsilon: {epsilon:.4f}")
            print(f"  Waiting Time: {avg_waiting_times[-1]:.2f} | Queue: {avg_queue_lengths[-1]:.2f}")
            print(f"  Avg Speed: {avg_speeds[-1]:.2f}")
            print(f"  Q-value: {q_value_means[-1]:.4f} ± {q_value_stds[-1]:.4f}")
            print(f"  Actions - NS: {action_counts[0]}, EW: {action_counts[1]}")
            
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
            try:
                traci.close()
            except:
                pass
            continue
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_td_errors': avg_td_errors,
        'avg_waiting_times': avg_waiting_times,
        'avg_queue_lengths': avg_queue_lengths,
        'avg_speeds': avg_speeds,
        'training_losses': training_losses,
        'epsilon_values': epsilon_values,
        'q_value_means': q_value_means,
        'q_value_stds': q_value_stds,
        'action_distributions': action_distributions
    }
    
    return q_network, results

# ==================== SARSA WITH FUNCTION APPROXIMATION ====================

def train_sarsa_function_approximation(num_episodes=100, gamma=0.99, 
                                       epsilon=0.1, epsilon_decay=0.995, 
                                       epsilon_min=0.01, max_steps=1500):
    """Train using SARSA with neural network function approximation"""
    
    # Initialize
    traci.start(["sumo", "-c", "rohit.sumocfg", "--start", "--quit-on-end"])
    state = get_state()
    input_dim = len(state)
    num_actions = 2
    traci.close()
    
    # Build Q-network
    q_network = build_value_function_network(input_dim, num_actions, learning_rate=0.001)
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    avg_td_errors = []
    avg_waiting_times = []
    training_losses = []
    epsilon_values = []
    
    print("="*70)
    print("SARSA WITH FUNCTION APPROXIMATION - TRAINING")
    print("="*70)
    
    for episode in range(num_episodes):
        try:
            traci.start(["sumo", "-c", "rohit.sumocfg", "--start", "--quit-on-end"])
            
            state = get_state()
            
            # Select first action
            if np.random.random() < epsilon:
                action = np.random.randint(num_actions)
            else:
                q_values = q_network.predict(state.reshape(1, -1), verbose=0)[0]
                action = np.argmax(q_values)
            
            episode_reward = 0
            episode_td_errors = []
            episode_waiting = []
            episode_losses = []
            steps = 0
            
            while traci.simulation.getMinExpectedNumber() > 0 and steps < max_steps:
                # Apply action
                apply_action(action)
                for _ in range(5):
                    traci.simulationStep()
                    steps += 1
                
                # Observe next state and reward
                next_state = get_state()
                reward = compute_reward()
                episode_reward += reward
                
                # Select next action (SARSA uses actual next action)
                if np.random.random() < epsilon:
                    next_action = np.random.randint(num_actions)
                else:
                    q_values_next = q_network.predict(next_state.reshape(1, -1), verbose=0)[0]
                    next_action = np.argmax(q_values_next)
                
                # Get Q-values
                q_values_current = q_network.predict(state.reshape(1, -1), verbose=0)[0]
                q_values_next_full = q_network.predict(next_state.reshape(1, -1), verbose=0)[0]
                
                # SARSA update: r + γ * Q(s', a') where a' is the actual next action
                td_target = reward + gamma * q_values_next_full[next_action]
                td_error = td_target - q_values_current[action]
                episode_td_errors.append(abs(td_error))
                
                # Update target
                target_q = q_values_current.copy()
                target_q[action] = td_target
                
                # Train
                history = q_network.fit(
                    state.reshape(1, -1),
                    target_q.reshape(1, -1),
                    verbose=0,
                    epochs=1
                )
                episode_losses.append(history.history['loss'][0])
                
                # Track metrics
                lane_ids = traci.lane.getIDList()
                episode_waiting.append(sum(traci.lane.getWaitingTime(lid) for lid in lane_ids))
                
                state = next_state
                action = next_action  # SARSA: move to next action
            
            traci.close()
            
            # Decay epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
            # Store metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            avg_td_errors.append(np.mean(episode_td_errors))
            avg_waiting_times.append(np.mean(episode_waiting))
            training_losses.append(np.mean(episode_losses))
            epsilon_values.append(epsilon)
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f} | TD Error: {avg_td_errors[-1]:.4f}")
            
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
            try:
                traci.close()
            except:
                pass
            continue
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_td_errors': avg_td_errors,
        'avg_waiting_times': avg_waiting_times,
        'training_losses': training_losses,
        'epsilon_values': epsilon_values
    }
    
    return q_network, results

# ==================== VISUALIZATION ====================

def plot_function_approximation_results(td0_results, sarsa_results=None):
    """Create comprehensive visualizations for function approximation"""
    
    if sarsa_results:
        fig = plt.figure(figsize=(20, 12))
        num_plots = 12
    else:
        fig = plt.figure(figsize=(18, 12))
        num_plots = 10
    
    # 1. Episode Rewards Comparison
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(td0_results['episode_rewards'], color='blue', alpha=0.6, 
             linewidth=1, label='TD(0)')
    if len(td0_results['episode_rewards']) >= 5:
        ma = np.convolve(td0_results['episode_rewards'], np.ones(5)/5, mode='valid')
        ax1.plot(range(2, len(ma)+2), ma, color='darkblue', linewidth=2, label='TD(0) MA(5)')
    
    if sarsa_results:
        ax1.plot(sarsa_results['episode_rewards'], color='red', alpha=0.6, 
                linewidth=1, label='SARSA')
        if len(sarsa_results['episode_rewards']) >= 5:
            ma_s = np.convolve(sarsa_results['episode_rewards'], np.ones(5)/5, mode='valid')
            ax1.plot(range(2, len(ma_s)+2), ma_s, color='darkred', linewidth=2, label='SARSA MA(5)')
    
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Total Reward', fontsize=11)
    ax1.set_title('Episode Rewards Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. TD Errors
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(td0_results['avg_td_errors'], color='green', linewidth=2, label='TD(0)')
    if sarsa_results:
        ax2.plot(sarsa_results['avg_td_errors'], color='orange', linewidth=2, label='SARSA')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Avg TD Error', fontsize=11)
    ax2.set_title('TD Error Convergence', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Loss
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(td0_results['training_losses'], color='red', linewidth=2, label='TD(0)')
    if sarsa_results:
        ax3.plot(sarsa_results['training_losses'], color='purple', linewidth=2, label='SARSA')
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Epsilon Decay
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(td0_results['epsilon_values'], color='purple', linewidth=2, label='TD(0)')
    if sarsa_results:
        ax4.plot(sarsa_results['epsilon_values'], color='brown', linewidth=2, 
                linestyle='--', label='SARSA')
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Epsilon', fontsize=11)
    ax4.set_title('Exploration Rate (Epsilon)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Waiting Times
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(td0_results['avg_waiting_times'], color='orange', linewidth=2, label='TD(0)')
    if sarsa_results:
        ax5.plot(sarsa_results['avg_waiting_times'], color='red', linewidth=2, label='SARSA')
    ax5.set_xlabel('Episode', fontsize=11)
    ax5.set_ylabel('Avg Waiting Time', fontsize=11)
    ax5.set_title('Average Waiting Time', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Queue Lengths
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(td0_results['avg_queue_lengths'], color='brown', linewidth=2)
    ax6.set_xlabel('Episode', fontsize=11)
    ax6.set_ylabel('Avg Queue Length', fontsize=11)
    ax6.set_title('Average Queue Length', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Average Speeds
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(td0_results['avg_speeds'], color='teal', linewidth=2)
    ax7.set_xlabel('Episode', fontsize=11)
    ax7.set_ylabel('Avg Speed (m/s)', fontsize=11)
    ax7.set_title('Average Vehicle Speed', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Q-Value Evolution
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(td0_results['q_value_means'], color='darkgreen', linewidth=2)
    if td0_results['q_value_stds']:
        means = np.array(td0_results['q_value_means'])
        stds = np.array(td0_results['q_value_stds'])
        ax8.fill_between(range(len(means)), means-stds, means+stds, alpha=0.3)
    ax8.set_xlabel('Episode', fontsize=11)
    ax8.set_ylabel('Q-Value', fontsize=11)
    ax8.set_title('Q-Value Evolution', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # 9. Episode Lengths
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(td0_results['episode_lengths'], color='navy', linewidth=2, label='TD(0)')
    if sarsa_results:
        ax9.plot(sarsa_results['episode_lengths'], color='maroon', linewidth=2, label='SARSA')
    ax9.set_xlabel('Episode', fontsize=11)
    ax9.set_ylabel('Steps', fontsize=11)
    ax9.set_title('Episode Lengths', fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Action Distribution
    ax10 = plt.subplot(3, 4, 10)
    ns_actions = [d[0] for d in td0_results['action_distributions']]
    ew_actions = [d[1] for d in td0_results['action_distributions']]
    ax10.plot(ns_actions, label='NS-Green', color='blue', linewidth=2)
    ax10.plot(ew_actions, label='EW-Green', color='red', linewidth=2)
    ax10.set_xlabel('Episode', fontsize=11)
    ax10.set_ylabel('Action Count', fontsize=11)
    ax10.set_title('Action Distribution', fontsize=12, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Reward Distribution
    ax11 = plt.subplot(3, 4, 11)
    ax11.hist(td0_results['episode_rewards'], bins=20, color='skyblue', 
             edgecolor='black', alpha=0.7, label='TD(0)')
    if sarsa_results:
        ax11.hist(sarsa_results['episode_rewards'], bins=20, color='salmon', 
                 edgecolor='black', alpha=0.5, label='SARSA')
    ax11.set_xlabel('Reward', fontsize=11)
    ax11.set_ylabel('Frequency', fontsize=11)
    ax11.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Performance Summary
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary = f"""
    FUNCTION APPROXIMATION
    TRAINING SUMMARY
    {'='*28}
    
    TD(0) RESULTS:
    Episodes: {len(td0_results['episode_rewards'])}
    Final Reward: {td0_results['episode_rewards'][-1]:.2f}
    Best Reward: {max(td0_results['episode_rewards']):.2f}
    Avg (Last 10): {np.mean(td0_results['episode_rewards'][-10:]):.2f}
    
    Final TD Error: {td0_results['avg_td_errors'][-1]:.4f}
    Final Loss: {td0_results['training_losses'][-1]:.4f}
    Final Waiting: {td0_results['avg_waiting_times'][-1]:.2f}
    Final Speed: {td0_results['avg_speeds'][-1]:.2f}
    Final Q-value: {td0_results['q_value_means'][-1]:.4f}
    """
    
    if sarsa_results:
        summary += f"""
    
    SARSA RESULTS:
    Final Reward: {sarsa_results['episode_rewards'][-1]:.2f}
    Best Reward: {max(sarsa_results['episode_rewards']):.2f}
    Avg (Last 10): {np.mean(sarsa_results['episode_rewards'][-10:]):.2f}
    """
    
    ax12.text(0.05, 0.5, summary, fontsize=8.5, family='monospace',
              verticalalignment='center')
    
    plt.tight_layout()
    
    if sarsa_results:
        plt.savefig('function_approximation_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'function_approximation_comparison.png'")
    else:
        plt.savefig('td0_function_approximation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'td0_function_approximation.png'")
    
    plt.show()

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FUNCTION APPROXIMATION WITH NEURAL NETWORKS")
    print("TEMPORAL DIFFERENCE LEARNING FOR TRAFFIC CONTROL")
    print("="*70 + "\n")
    
    mode = input("Select mode (1: TD(0) only, 2: Both TD(0) and SARSA): ").strip()
    
    try:
        # Train TD(0) with Function Approximation
        print("\n" + "="*70)
        print("TRAINING TD(0) WITH FUNCTION APPROXIMATION")
        print("="*70)
        td0_network, td0_results = train_td0_function_approximation(
            num_episodes=50,
            gamma=0.99,
            epsilon=0.2,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        # Save TD(0) model
        td0_network.save('td0_function_approx_model.h5')
        print("\n✓ TD(0) model saved: td0_function_approx_model.h5")
        
        sarsa_results = None
        if mode == '2':
            # Train SARSA with Function Approximation
            print("\n" + "="*70)
            print("TRAINING SARSA WITH FUNCTION APPROXIMATION")
            print("="*70)
            sarsa_network, sarsa_results = train_sarsa_function_approximation(
                num_episodes=50,
                gamma=0.99,
                epsilon=0.2,
                epsilon_decay=0.995,
                epsilon_min=0.01
            )
            
            # Save SARSA model
            sarsa_network.save('sarsa_function_approx_model.h5')
            print("\n✓ SARSA model saved: sarsa_function_approx_model.h5")
        
        # Plot results
        plot_function_approximation_results(td0_results, sarsa_results)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        
        # Print final comparison
        print("\n" + "="*70)
        print("FINAL PERFORMANCE COMPARISON")
        print("="*70)
        print(f"\nTD(0) Performance:")
        print(f"  Average Reward (Last 10): {np.mean(td0_results['episode_rewards'][-10:]):.2f}")
        print(f"  Best Reward: {max(td0_results['episode_rewards']):.2f}")
        print(f"  Final TD Error: {td0_results['avg_td_errors'][-1]:.4f}")
        print(f"  Final Waiting Time: {td0_results['avg_waiting_times'][-1]:.2f}")
        
        if sarsa_results:
            print(f"\nSARSA Performance:")
            print(f"  Average Reward (Last 10): {np.mean(sarsa_results['episode_rewards'][-10:]):.2f}")
            print(f"  Best Reward: {max(sarsa_results['episode_rewards']):.2f}")
            print(f"  Final TD Error: {sarsa_results['avg_td_errors'][-1]:.4f}")
            print(f"  Final Waiting Time: {sarsa_results['avg_waiting_times'][-1]:.2f}")
            
            # Determine winner
            td0_avg = np.mean(td0_results['episode_rewards'][-10:])
            sarsa_avg = np.mean(sarsa_results['episode_rewards'][-10:])
            
            print("\n" + "="*70)
            if td0_avg > sarsa_avg:
                improvement = ((td0_avg - sarsa_avg) / abs(sarsa_avg) * 100)
                print(f"🏆 WINNER: TD(0) (Better by {improvement:.2f}%)")
            else:
                improvement = ((sarsa_avg - td0_avg) / abs(td0_avg) * 100)
                print(f"🏆 WINNER: SARSA (Better by {improvement:.2f}%)")
            print("="*70)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            traci.close()
        except:
            pass