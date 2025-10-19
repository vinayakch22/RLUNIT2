import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# ==================== STATE AND ACTION UTILITIES ====================

def discretize_state(state, bins=10):
    """Discretize continuous state into bins for tabular methods"""
    discretized = []
    for val in state:
        # Normalize and bin the values with better granularity
        if val < 0:
            val = 0
        # Use better binning: 0-5, 5-10, 10-20, 20-30, 30-50, 50-100, 100+
        if val < 5:
            bin_idx = 0
        elif val < 10:
            bin_idx = 1
        elif val < 20:
            bin_idx = 2
        elif val < 30:
            bin_idx = 3
        elif val < 50:
            bin_idx = 4
        elif val < 100:
            bin_idx = 5
        else:
            bin_idx = min(int(val / 50), bins - 1)
        discretized.append(bin_idx)
    return tuple(discretized)

def get_state():
    """Get current state from SUMO"""
    try:
        lane_ids = traci.lane.getIDList()
        if not lane_ids:
            return np.array([0])
        
        total_waiting_times = [traci.lane.getWaitingTime(lane_id) for lane_id in lane_ids]
        
        # Get time since last phase change
        try:
            time_since_change = traci.trafficlight.getPhase("B2")
        except:
            time_since_change = 0
            
        return np.array(total_waiting_times + [time_since_change])
    except Exception as e:
        print(f"Error in get_state: {e}")
        return np.array([0])

def compute_reward():
    """Compute reward based on waiting times"""
    try:
        lane_ids = traci.lane.getIDList()
        if not lane_ids:
            return 0
        
        total_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in lane_ids)
        
        # Penalize by waiting time
        return -total_waiting_time
    except Exception as e:
        print(f"Error in compute_reward: {e}")
        return 0

def apply_action(action, min_duration=5):
    """Apply action to traffic light with minimum phase duration"""
    try:
        current_phase = traci.trafficlight.getPhase("B2")
        phase_duration = traci.trafficlight.getPhaseDuration("B2")
        
        # Map actions to phases: 0=NS-Green (phase 0), 1=EW-Green (phase 2)
        target_phase = 0 if action == 0 else 2
        
        # Only change if different and minimum duration passed
        if current_phase != target_phase and phase_duration >= min_duration:
            traci.trafficlight.setPhase("B2", target_phase)
            traci.trafficlight.setPhaseDuration("B2", 30)  # Set green duration
    except Exception as e:
        print(f"Error in apply_action: {e}")

# ==================== POLICY ITERATION ====================

class PolicyIteration:
    def __init__(self, num_actions=2, gamma=0.99, theta=0.01):
        self.num_actions = num_actions
        self.gamma = gamma
        self.theta = theta  # Convergence threshold
        self.V = defaultdict(float)  # State value function
        self.policy = defaultdict(lambda: 0)  # Deterministic policy
        self.transitions = defaultdict(lambda: defaultdict(list))  # s,a -> [(s', r, prob)]
        
    def collect_transitions(self, num_episodes=30):
        """Collect transition data from environment"""
        print("Collecting transition data for Policy Iteration...")
        transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        reward_sums = defaultdict(lambda: defaultdict(float))
        reward_counts = defaultdict(lambda: defaultdict(int))
        
        for episode in range(num_episodes):
            print(f"Data collection episode {episode + 1}/{num_episodes}")
            
            try:
                traci.load(["-c", "rohit.sumocfg"])
                step_count = 0
                max_steps = 1500  # Increased max steps
                action_change_interval = 10  # Change action every 10 steps
                
                while traci.simulation.getMinExpectedNumber() > 0 and step_count < max_steps:
                    state = discretize_state(get_state())
                    
                    # Change action periodically to explore more
                    if step_count % action_change_interval == 0:
                        action = np.random.randint(self.num_actions)
                    
                    apply_action(action)
                    
                    # Take multiple simulation steps per action
                    for _ in range(5):
                        traci.simulationStep()
                        step_count += 1
                    
                    next_state = discretize_state(get_state())
                    reward = compute_reward()
                    
                    # Count transitions
                    transition_counts[state][action][next_state] += 1
                    reward_sums[state][action] += reward
                    reward_counts[state][action] += 1
                    
            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")
                continue
        
        # Convert counts to probabilities
        for s in transition_counts:
            for a in transition_counts[s]:
                total = sum(transition_counts[s][a].values())
                if total > 0:
                    for s_next in transition_counts[s][a]:
                        prob = transition_counts[s][a][s_next] / total
                        avg_reward = reward_sums[s][a] / reward_counts[s][a] if reward_counts[s][a] > 0 else 0
                        self.transitions[s][a].append((s_next, avg_reward, prob))
        
        print(f"Collected {len(self.transitions)} unique states")
        print(f"Total state-action pairs: {sum(len(self.transitions[s]) for s in self.transitions)}")
    
    def policy_evaluation(self):
        """Evaluate current policy"""
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            delta = 0
            for state in self.transitions:
                v = self.V[state]
                action = self.policy[state]
                
                # Bellman expectation equation
                new_v = 0
                if action in self.transitions[state]:
                    for next_state, reward, prob in self.transitions[state][action]:
                        new_v += prob * (reward + self.gamma * self.V[next_state])
                
                self.V[state] = new_v
                delta = max(delta, abs(v - new_v))
            
            iteration += 1
            if delta < self.theta:
                print(f"Policy evaluation converged in {iteration} iterations (delta={delta:.6f})")
                break
        
        if iteration >= max_iterations:
            print(f"Policy evaluation stopped at max iterations: {max_iterations}")
    
    def policy_improvement(self):
        """Improve policy based on current value function"""
        policy_stable = True
        
        for state in self.transitions:
            old_action = self.policy[state]
            
            # Find best action
            action_values = []
            for action in range(self.num_actions):
                if action in self.transitions[state]:
                    value = 0
                    for next_state, reward, prob in self.transitions[state][action]:
                        value += prob * (reward + self.gamma * self.V[next_state])
                    action_values.append(value)
                else:
                    action_values.append(float('-inf'))
            
            if action_values:
                best_action = np.argmax(action_values)
                self.policy[state] = best_action
                
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable
    
    def train(self, max_iterations=50):
        """Run policy iteration algorithm"""
        print("\n=== Starting Policy Iteration ===")
        self.collect_transitions()
        
        if not self.transitions:
            print("ERROR: No transitions collected!")
            return self.policy, self.V
        
        policy_history = []
        
        for iteration in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"Policy Iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*50}")
            
            # Store old policy for comparison
            old_policy = dict(self.policy)
            
            # Policy Evaluation
            self.policy_evaluation()
            
            # Policy Improvement
            policy_stable = self.policy_improvement()
            
            # Track policy changes
            num_changes = sum(1 for s in self.policy if old_policy.get(s, -1) != self.policy[s])
            policy_history.append({
                'iteration': iteration + 1,
                'policy': dict(self.policy),
                'values': dict(self.V),
                'num_changes': num_changes,
                'num_states': len(self.policy)
            })
            
            print(f"States with policy changes: {num_changes}/{len(self.policy)}")
            
            # Print current policy summary
            action_counts = {0: 0, 1: 0}
            for state, action in self.policy.items():
                action_counts[action] += 1
            print(f"Action Distribution: NS-Green={action_counts[0]}, EW-Green={action_counts[1]}")
            
            if policy_stable:
                print(f"\n✓ Policy converged in {iteration + 1} iterations!")
                break
        
        self.policy_history = policy_history
        return self.policy, self.V

# ==================== VALUE ITERATION ====================

class ValueIteration:
    def __init__(self, num_actions=2, gamma=0.99, theta=0.01):
        self.num_actions = num_actions
        self.gamma = gamma
        self.theta = theta
        self.V = defaultdict(float)
        self.policy = defaultdict(lambda: 0)
        self.transitions = defaultdict(lambda: defaultdict(list))
    
    def collect_transitions(self, num_episodes=30):
        """Collect transition data from environment"""
        print("Collecting transition data for Value Iteration...")
        transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        reward_sums = defaultdict(lambda: defaultdict(float))
        reward_counts = defaultdict(lambda: defaultdict(int))
        
        for episode in range(num_episodes):
            print(f"Data collection episode {episode + 1}/{num_episodes}")
            
            try:
                traci.load(["-c", "rohit.sumocfg"])
                step_count = 0
                max_steps = 1500
                action_change_interval = 10
                
                while traci.simulation.getMinExpectedNumber() > 0 and step_count < max_steps:
                    state = discretize_state(get_state())
                    
                    if step_count % action_change_interval == 0:
                        action = np.random.randint(self.num_actions)
                    
                    apply_action(action)
                    
                    for _ in range(5):
                        traci.simulationStep()
                        step_count += 1
                    
                    next_state = discretize_state(get_state())
                    reward = compute_reward()
                    
                    transition_counts[state][action][next_state] += 1
                    reward_sums[state][action] += reward
                    reward_counts[state][action] += 1
                    
            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")
                continue
        
        # Convert to probabilities
        for s in transition_counts:
            for a in transition_counts[s]:
                total = sum(transition_counts[s][a].values())
                if total > 0:
                    for s_next in transition_counts[s][a]:
                        prob = transition_counts[s][a][s_next] / total
                        avg_reward = reward_sums[s][a] / reward_counts[s][a] if reward_counts[s][a] > 0 else 0
                        self.transitions[s][a].append((s_next, avg_reward, prob))
        
        print(f"Collected {len(self.transitions)} unique states")
        print(f"Total state-action pairs: {sum(len(self.transitions[s]) for s in self.transitions)}")
    
    def train(self, max_iterations=100):
        """Run value iteration algorithm"""
        print("\n=== Starting Value Iteration ===")
        self.collect_transitions()
        
        if not self.transitions:
            print("ERROR: No transitions collected!")
            return self.policy, self.V, []
        
        value_deltas = []
        value_history = []
        
        for iteration in range(max_iterations):
            delta = 0
            
            for state in self.transitions:
                v = self.V[state]
                
                # Bellman optimality equation
                action_values = []
                for action in range(self.num_actions):
                    if action in self.transitions[state]:
                        value = 0
                        for next_state, reward, prob in self.transitions[state][action]:
                            value += prob * (reward + self.gamma * self.V[next_state])
                        action_values.append(value)
                    else:
                        action_values.append(float('-inf'))
                
                if action_values:
                    self.V[state] = max(action_values)
                    delta = max(delta, abs(v - self.V[state]))
            
            value_deltas.append(delta)
            value_history.append({
                'iteration': iteration + 1,
                'delta': delta,
                'values': dict(self.V)
            })
            
            if iteration % 10 == 0 or delta < self.theta:
                print(f"Iteration {iteration + 1:3d}, Delta: {delta:.8f}")
            
            if delta < self.theta:
                print(f"\n✓ Value iteration converged in {iteration + 1} iterations!")
                break
        
        # Extract optimal policy
        print("\nExtracting optimal policy...")
        for state in self.transitions:
            action_values = []
            for action in range(self.num_actions):
                if action in self.transitions[state]:
                    value = 0
                    for next_state, reward, prob in self.transitions[state][action]:
                        value += prob * (reward + self.gamma * self.V[next_state])
                    action_values.append(value)
                else:
                    action_values.append(float('-inf'))
            
            if action_values:
                self.policy[state] = np.argmax(action_values)
        
        # Print final policy summary
        action_counts = {0: 0, 1: 0}
        for state, action in self.policy.items():
            action_counts[action] += 1
        print(f"Final Action Distribution: NS-Green={action_counts[0]}, EW-Green={action_counts[1]}")
        
        self.value_history = value_history
        return self.policy, self.V, value_deltas

# ==================== POLICY DISPLAY FUNCTIONS ====================

def print_policy_details(policy, values, method_name):
    """Print detailed policy information"""
    print("\n" + "="*70)
    print(f"{method_name.upper()} - POLICY DETAILS")
    print("="*70)
    
    action_names = {0: "NS-Green (North-South)", 1: "EW-Green (East-West)"}
    
    # Sort states for consistent display
    sorted_states = sorted(policy.keys())
    
    print(f"\nTotal States: {len(sorted_states)}")
    print(f"Total Actions: 2 (NS-Green, EW-Green)")
    print("\n" + "-"*70)
    print(f"{'State':<30} {'Action':<25} {'Value':<15}")
    print("-"*70)
    
    for state in sorted_states[:20]:  # Show first 20 states
        action = policy[state]
        value = values.get(state, 0)
        print(f"{str(state):<30} {action_names[action]:<25} {value:>14.4f}")
    
    if len(sorted_states) > 20:
        print(f"... ({len(sorted_states) - 20} more states)")
    
    # Action statistics
    action_counts = {0: 0, 1: 0}
    for action in policy.values():
        action_counts[action] += 1
    
    print("\n" + "-"*70)
    print("ACTION DISTRIBUTION:")
    print("-"*70)
    total = sum(action_counts.values())
    for action, count in action_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{action_names[action]:<30} {count:>6} states ({percentage:>5.1f}%)")
    
    # Value statistics
    if values:
        value_list = list(values.values())
        print("\n" + "-"*70)
        print("VALUE FUNCTION STATISTICS:")
        print("-"*70)
        print(f"{'Mean Value:':<30} {np.mean(value_list):>14.4f}")
        print(f"{'Std Dev:':<30} {np.std(value_list):>14.4f}")
        print(f"{'Min Value:':<30} {np.min(value_list):>14.4f}")
        print(f"{'Max Value:':<30} {np.max(value_list):>14.4f}")
    
    print("="*70 + "\n")

def print_policy_iteration_history(pi_agent):
    """Print policy iteration history"""
    if not hasattr(pi_agent, 'policy_history'):
        return
    
    print("\n" + "="*70)
    print("POLICY ITERATION - CONVERGENCE HISTORY")
    print("="*70)
    print(f"{'Iteration':<12} {'States':<10} {'Changes':<12} {'NS-Green %':<15} {'EW-Green %':<15}")
    print("-"*70)
    
    for hist in pi_agent.policy_history:
        policy = hist['policy']
        action_counts = {0: 0, 1: 0}
        for action in policy.values():
            action_counts[action] += 1
        total = sum(action_counts.values())
        
        ns_pct = (action_counts[0] / total * 100) if total > 0 else 0
        ew_pct = (action_counts[1] / total * 100) if total > 0 else 0
        
        print(f"{hist['iteration']:<12} {hist['num_states']:<10} {hist['num_changes']:<12} "
              f"{ns_pct:>13.1f}% {ew_pct:>14.1f}%")
    
    print("="*70 + "\n")

def compare_policies(pi_policy, vi_policy, pi_values, vi_values):
    """Compare two policies"""
    print("\n" + "="*70)
    print("POLICY COMPARISON: POLICY ITERATION vs VALUE ITERATION")
    print("="*70)
    
    # Find common states
    common_states = set(pi_policy.keys()) & set(vi_policy.keys())
    pi_only_states = set(pi_policy.keys()) - set(vi_policy.keys())
    vi_only_states = set(vi_policy.keys()) - set(pi_policy.keys())
    
    print(f"\nCommon states: {len(common_states)}")
    print(f"Only in Policy Iteration: {len(pi_only_states)}")
    print(f"Only in Value Iteration: {len(vi_only_states)}")
    
    # Compare actions on common states
    same_action = 0
    different_action = 0
    action_names = {0: "NS-Green", 1: "EW-Green"}
    
    print("\n" + "-"*70)
    print("AGREEMENT ON COMMON STATES:")
    print("-"*70)
    
    differences = []
    for state in sorted(common_states)[:30]:  # Show first 30
        pi_action = pi_policy[state]
        vi_action = vi_policy[state]
        
        if pi_action == vi_action:
            same_action += 1
        else:
            different_action += 1
            pi_val = pi_values.get(state, 0)
            vi_val = vi_values.get(state, 0)
            differences.append((state, pi_action, vi_action, pi_val, vi_val))
    
    agreement_pct = (same_action / len(common_states) * 100) if common_states else 0
    
    print(f"Agreement: {same_action}/{len(common_states)} states ({agreement_pct:.1f}%)")
    print(f"Disagreement: {different_action}/{len(common_states)} states ({100-agreement_pct:.1f}%)")
    
    if differences:
        print("\n" + "-"*70)
        print("SAMPLE DISAGREEMENTS:")
        print("-"*70)
        print(f"{'State':<25} {'PI Action':<15} {'VI Action':<15} {'PI Value':<12} {'VI Value':<12}")
        print("-"*70)
        for state, pi_act, vi_act, pi_val, vi_val in differences[:10]:
            print(f"{str(state):<25} {action_names[pi_act]:<15} {action_names[vi_act]:<15} "
                  f"{pi_val:>11.4f} {vi_val:>11.4f}")
    
    print("="*70 + "\n")

def determine_best_policy(pi_rewards, vi_rewards, pi_policy, vi_policy, pi_values, vi_values):
    """Determine and display the best policy"""
    pi_avg = np.mean(pi_rewards)
    vi_avg = np.mean(vi_rewards)
    
    print("\n" + "="*70)
    print("🏆 BEST POLICY DETERMINATION")
    print("="*70)
    
    print("\nPERFORMANCE METRICS:")
    print("-"*70)
    print(f"{'Method':<30} {'Avg Reward':<15} {'Std Dev':<15}")
    print("-"*70)
    print(f"{'Policy Iteration':<30} {pi_avg:>14.2f} {np.std(pi_rewards):>14.2f}")
    print(f"{'Value Iteration':<30} {vi_avg:>14.2f} {np.std(vi_rewards):>14.2f}")
    print("-"*70)
    
    if pi_avg > vi_avg:
        winner = "Policy Iteration"
        best_policy = pi_policy
        best_values = pi_values
        best_rewards = pi_rewards
        improvement = ((pi_avg - vi_avg) / abs(vi_avg) * 100) if vi_avg != 0 else 0
    else:
        winner = "Value Iteration"
        best_policy = vi_policy
        best_values = vi_values
        best_rewards = vi_rewards
        improvement = ((vi_avg - pi_avg) / abs(pi_avg) * 100) if pi_avg != 0 else 0
    
    print(f"\n🏆 WINNER: {winner}")
    print(f"Performance improvement: {abs(improvement):.2f}%")
    
    print("\n" + "="*70)
    print(f"BEST POLICY DETAILS ({winner})")
    print("="*70)
    
    action_names = {0: "NS-Green (North-South)", 1: "EW-Green (East-West)"}
    
    # Show top states by value
    sorted_by_value = sorted(best_values.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTOP 15 STATES BY VALUE:")
    print("-"*70)
    print(f"{'Rank':<6} {'State':<25} {'Action':<20} {'Value':<15}")
    print("-"*70)
    
    for i, (state, value) in enumerate(sorted_by_value[:15], 1):
        action = best_policy[state]
        print(f"{i:<6} {str(state):<25} {action_names[action]:<20} {value:>14.4f}")
    
    # Show action distribution
    action_counts = {0: 0, 1: 0}
    for action in best_policy.values():
        action_counts[action] += 1
    
    print("\n" + "-"*70)
    print("BEST POLICY ACTION DISTRIBUTION:")
    print("-"*70)
    total = sum(action_counts.values())
    for action, count in action_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"{action_names[action]:<30} {count:>6} states ({percentage:>5.1f}%) {bar}")
    
    print("\n" + "-"*70)
    print("REWARD STATISTICS (Best Policy):")
    print("-"*70)
    print(f"{'Mean Reward:':<30} {np.mean(best_rewards):>14.2f}")
    print(f"{'Median Reward:':<30} {np.median(best_rewards):>14.2f}")
    print(f"{'Std Deviation:':<30} {np.std(best_rewards):>14.2f}")
    print(f"{'Min Reward:':<30} {np.min(best_rewards):>14.2f}")
    print(f"{'Max Reward:':<30} {np.max(best_rewards):>14.2f}")
    
    print("="*70 + "\n")
    
    return winner, best_policy, best_values

# ==================== EVALUATION ====================

def evaluate_policy(policy, num_episodes=10):
    """Evaluate a learned policy"""
    print("\n=== Evaluating Policy ===")
    total_rewards = []
    total_waiting_times = []
    
    for episode in range(num_episodes):
        print(f"Evaluation episode {episode + 1}/{num_episodes}")
        
        try:
            traci.load(["-c", "rohit.sumocfg"])
            episode_reward = 0
            episode_waiting = 0
            step_count = 0
            max_steps = 1500
            action_count = {0: 0, 1: 0}
            
            while traci.simulation.getMinExpectedNumber() > 0 and step_count < max_steps:
                state = discretize_state(get_state())
                action = policy.get(state, 0)  # Default to action 0 if state not seen
                action_count[action] += 1
                
                apply_action(action)
                
                # Take multiple steps per action
                for _ in range(5):
                    traci.simulationStep()
                    step_count += 1
                    reward = compute_reward()
                    episode_reward += reward
                    
                    # Track waiting time
                    waiting = sum(traci.lane.getWaitingTime(lane_id) for lane_id in traci.lane.getIDList())
                    episode_waiting += waiting
            
            total_rewards.append(episode_reward)
            total_waiting_times.append(episode_waiting)
            print(f"Episode reward: {episode_reward:.2f}, Total waiting time: {episode_waiting:.2f}")
            print(f"Action usage - NS: {action_count[0]}, EW: {action_count[1]}")
            
        except Exception as e:
            print(f"Error in evaluation episode {episode + 1}: {e}")
            total_rewards.append(0)
            total_waiting_times.append(0)
    
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    avg_waiting = np.mean(total_waiting_times) if total_waiting_times else 0
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"Average total waiting time: {avg_waiting:.2f}")
    return total_rewards

# ==================== VISUALIZATION ====================

def plot_comparison(pi_rewards, vi_rewards, vi_deltas=None):
    """Plot comparison of both methods"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Evaluation rewards comparison
    axes[0].plot(pi_rewards, label="Policy Iteration", marker='o', linewidth=2, color='blue')
    axes[0].plot(vi_rewards, label="Value Iteration", marker='s', linewidth=2, color='red')
    axes[0].set_xlabel("Episode", fontsize=14)
    axes[0].set_ylabel("Total Reward", fontsize=14)
    axes[0].set_title("Policy Evaluation: Reward Comparison", fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Value iteration convergence
    if vi_deltas and len(vi_deltas) > 0:
        axes[1].plot(vi_deltas, color='green', linewidth=2)
        axes[1].set_xlabel("Iteration", fontsize=14)
        axes[1].set_ylabel("Max Value Change (Delta)", fontsize=14)
        axes[1].set_title("Value Iteration Convergence", fontsize=16)
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No convergence data available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[1].transAxes, fontsize=14)
    
    plt.tight_layout()
    plt.savefig('policy_value_iteration_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'policy_value_iteration_comparison.png'")
    plt.show()

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    sumo_cfg_file = "rohit.sumocfg"
    
    # Check if config file exists
    if not os.path.exists(sumo_cfg_file):
        print(f"ERROR: Configuration file '{sumo_cfg_file}' not found!")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Start SUMO
    try:
        print("Starting SUMO...")
        traci.start(["sumo-gui", "-c", sumo_cfg_file, "--start", "--quit-on-end"])
        print("SUMO started successfully!")
    except Exception as e:
        print(f"Error starting SUMO: {e}")
        print("Trying without GUI...")
        try:
            traci.start(["sumo", "-c", sumo_cfg_file, "--start", "--quit-on-end"])
            print("SUMO started successfully (no GUI)!")
        except Exception as e2:
            print(f"Error starting SUMO: {e2}")
            sys.exit(1)
    
    try:
        # Policy Iteration
        print("\n" + "="*60)
        print("TRAINING WITH POLICY ITERATION")
        print("="*60)
        pi_agent = PolicyIteration(num_actions=2, gamma=0.99, theta=0.1)
        pi_policy, pi_values = pi_agent.train(max_iterations=20)
        
        # Print Policy Iteration details
        print_policy_details(pi_policy, pi_values, "Policy Iteration")
        print_policy_iteration_history(pi_agent)
        
        # Evaluate Policy Iteration
        pi_rewards = evaluate_policy(pi_policy, num_episodes=10)
        
        # Value Iteration
        print("\n" + "="*60)
        print("TRAINING WITH VALUE ITERATION")
        print("="*60)
        vi_agent = ValueIteration(num_actions=2, gamma=0.99, theta=0.1)
        vi_policy, vi_values, vi_deltas = vi_agent.train(max_iterations=100)
        
        # Print Value Iteration details
        print_policy_details(vi_policy, vi_values, "Value Iteration")
        
        # Evaluate Value Iteration
        vi_rewards = evaluate_policy(vi_policy, num_episodes=10)
        
        # Compare both policies
        compare_policies(pi_policy, vi_policy, pi_values, vi_values)
        
        # Determine and display best policy
        winner, best_policy, best_values = determine_best_policy(
            pi_rewards, vi_rewards, pi_policy, vi_policy, pi_values, vi_values
        )
        
        # Final Comparison Summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"Policy Iteration - Avg Reward: {np.mean(pi_rewards):.2f}")
        print(f"Value Iteration  - Avg Reward: {np.mean(vi_rewards):.2f}")
        print(f"Policy Iteration - Num States: {len(pi_policy)}")
        print(f"Value Iteration  - Num States: {len(vi_policy)}")
        print(f"\n🏆 Best Method: {winner}")
        print("="*60)
        
        # Plot results
        plot_comparison(pi_rewards, vi_rewards, vi_deltas)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            traci.close()
            print("\nSUMO closed successfully")
        except:
            pass