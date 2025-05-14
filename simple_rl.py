import numpy as np
import pandas as pd
import random
from datetime import datetime
import json
import os

# Directory for storing models
MODELS_DIR = "models"

class TradingRLAgent:
    """
    A simple reinforcement learning agent for trading using Q-learning
    """
    
    def __init__(self, state_size=10, action_size=3, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize the agent with parameters
        
        Parameters:
        -----------
        state_size: int
            Size of the state representation
        action_size: int
            Number of possible actions (0: hold, 1: buy, 2: sell)
        learning_rate: float
            Learning rate for updating Q-values
        gamma: float
            Discount factor for future rewards
        epsilon: float
            Exploration rate (probability of taking a random action)
        epsilon_decay: float
            Rate at which epsilon decreases over time
        epsilon_min: float
            Minimum value of epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = []  # replay memory
        
        # Initialize Q-table with zeros
        self.q_table = {}
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Choose an action based on the current state
        """
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to string for dictionary lookup
        state_key = self._get_state_key(state)
        
        # If state not in Q-table, initialize with zeros
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def _get_state_key(self, state):
        """
        Convert state array to a string key for Q-table
        """
        # Round state values to reduce state space
        rounded_state = np.round(state, 2)
        return str(rounded_state.tolist())
    
    def replay(self, batch_size=32):
        """
        Train the agent on stored experiences
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample random experiences from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_key = self._get_state_key(state)
            next_state_key = self._get_state_key(next_state)
            
            # Initialize Q-values if not in table
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_table[next_state_key])
            
            # Update Q-value
            self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, name):
        """
        Save the model to a file
        """
        # Ensure directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Convert Q-table keys from string back to lists for saving
        q_table_serializable = {}
        for key, value in self.q_table.items():
            q_table_serializable[key] = value.tolist()
        
        # Create model data
        model_data = {
            "q_table": q_table_serializable,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "saved_date": datetime.now().isoformat()
        }
        
        # Save to file
        filename = f"{name}.json"
        filepath = os.path.join(MODELS_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
            
        print(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a model from a file
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Create a new agent with the saved parameters
        agent = cls(
            state_size=model_data.get("state_size", 10),
            action_size=model_data.get("action_size", 3),
            learning_rate=model_data.get("learning_rate", 0.001),
            gamma=model_data.get("gamma", 0.95),
            epsilon=model_data.get("epsilon", 0.01),  # Use saved exploration rate
            epsilon_decay=model_data.get("epsilon_decay", 0.995),
            epsilon_min=model_data.get("epsilon_min", 0.01)
        )
        
        # Load Q-table
        q_table = model_data.get("q_table", {})
        for key, value in q_table.items():
            agent.q_table[key] = np.array(value)
        
        print(f"Model loaded from {filepath}")
        return agent


class TradingEnvironment:
    """
    A simple trading environment for reinforcement learning
    """
    
    def __init__(self, data, initial_balance=10000, transaction_fee=0.001):
        """
        Initialize the environment
        
        Parameters:
        -----------
        data: pandas.DataFrame
            Market data with OHLCV and indicators
        initial_balance: float
            Initial account balance
        transaction_fee: float
            Transaction fee as a percentage (0.001 = 0.1%)
        """
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reset()
    
    def reset(self):
        """
        Reset the environment to the initial state
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.shares_value = 0
        self.total_sales_value = 0
        self.total_buy_value = 0
        self.trades = []
        self.current_price = self.data.iloc[self.current_step]['Close']
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get the current state representation
        """
        # Get market data for current position
        market_data = self.data.iloc[self.current_step]
        
        # Extract basic features
        close = market_data['Close']
        
        # Extract indicators if available
        rsi = market_data.get('RSI', 50)  # Default to 50 if not available
        
        # Previous price changes (returns)
        if self.current_step > 0:
            prev_close = self.data.iloc[self.current_step-1]['Close']
            price_change = (close - prev_close) / prev_close
        else:
            price_change = 0
        
        # Moving averages if available
        ma_20 = market_data.get('MA_20', close)
        ma_50 = market_data.get('MA_50', close)
        ma_ratio = ma_20 / ma_50 if ma_50 != 0 else 1.0
        
        # Position features
        position_value = self.shares_held * close
        portfolio_value = self.balance + position_value
        profit_pct = (portfolio_value / self.initial_balance - 1) * 100
        
        # Create state array
        state = np.array([
            price_change,
            rsi / 100,  # Normalize to 0-1
            ma_ratio - 1,  # Normalize around 0
            self.shares_held > 0,  # Boolean: Do we have a position?
            position_value / portfolio_value if portfolio_value > 0 else 0,  # Position size
            profit_pct / 100  # Normalize profit percentage
        ])
        
        return state
    
    def step(self, action):
        """
        Take an action in the environment
        
        Parameters:
        -----------
        action: int
            0: Hold, 1: Buy, 2: Sell
            
        Returns:
        --------
        tuple
            (next_state, reward, done, info)
        """
        # Current values before action
        current_price = self.data.iloc[self.current_step]['Close']
        prev_portfolio_value = self.balance + (self.shares_held * current_price)
        
        # Initialize trade tracking variables
        trade_completed = False
        trade_info = None
        
        # Execute action
        if action == 1:  # Buy
            # Calculate max shares that can be bought
            max_shares = self.balance / (current_price * (1 + self.transaction_fee))
            # Buy all possible shares
            shares_bought = max_shares
            cost = shares_bought * current_price * (1 + self.transaction_fee)
            
            # Update
            self.balance -= cost
            self.shares_held += shares_bought
            self.total_buy_value += shares_bought * current_price
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'type': 'buy',
                'shares': shares_bought,
                'price': current_price,
                'cost': cost,
                'date': self.data.index[self.current_step] if hasattr(self.data.index, '__getitem__') else None
            })
            
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # Check for paired trades (buy followed by sell) to calculate trade profit
                for i in range(len(self.trades)-1, -1, -1):
                    if self.trades[i]['type'] == 'buy':
                        # Found matching buy trade
                        buy_price = self.trades[i]['price']
                        sell_price = current_price
                        # Calculate profit percentage
                        profit_pct = (sell_price - buy_price) / buy_price * 100
                        trade_completed = True
                        trade_info = {
                            'entry_date': self.trades[i].get('date'),
                            'exit_date': self.data.index[self.current_step] if hasattr(self.data.index, '__getitem__') else None,
                            'entry_price': buy_price,
                            'exit_price': sell_price,
                            'profit_pct': profit_pct,
                            'type': 'long'
                        }
                        break
                
                # Sell all shares
                sale_value = self.shares_held * current_price * (1 - self.transaction_fee)
                
                # Update
                self.balance += sale_value
                self.total_sales_value += self.shares_held * current_price
                self.shares_held = 0
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'shares': self.shares_held,
                    'price': current_price,
                    'value': sale_value,
                    'date': self.data.index[self.current_step] if hasattr(self.data.index, '__getitem__') else None
                })
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        if done:
            # If done, sell any remaining shares
            if self.shares_held > 0:
                # Check for paired trades for the final sell
                for i in range(len(self.trades)-1, -1, -1):
                    if self.trades[i]['type'] == 'buy':
                        # Found matching buy trade
                        buy_price = self.trades[i]['price']
                        sell_price = current_price
                        # Calculate profit percentage
                        profit_pct = (sell_price - buy_price) / buy_price * 100
                        trade_completed = True
                        trade_info = {
                            'entry_date': self.trades[i].get('date'),
                            'exit_date': self.data.index[self.current_step] if hasattr(self.data.index, '__getitem__') else None,
                            'entry_price': buy_price,
                            'exit_price': sell_price,
                            'profit_pct': profit_pct,
                            'type': 'long'
                        }
                        break
                
                sale_value = self.shares_held * current_price * (1 - self.transaction_fee)
                self.balance += sale_value
                self.total_sales_value += self.shares_held * current_price
                self.shares_held = 0
        
        # Calculate reward
        current_price = self.data.iloc[self.current_step]['Close']
        current_portfolio_value = self.balance + (self.shares_held * current_price)
        reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value * 100
        
        # Get next state
        next_state = self._get_state()
        
        # Information dictionary
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'trade_completed': trade_completed
        }
        
        # Add trade info if a trade was completed
        if trade_completed and trade_info:
            info['trade'] = trade_info
        
        return next_state, reward, done, info
    
    def render(self):
        """
        Render the current state of the environment
        """
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_value = self.balance + (self.shares_held * current_price)
        profit = portfolio_value - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        
        info = {
            'step': self.current_step,
            'date': self.data.index[self.current_step],
            'price': current_price,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'position_value': self.shares_held * current_price,
            'portfolio_value': portfolio_value,
            'profit': profit,
            'profit_pct': profit_pct
        }
        
        return info


def train_agent(data, episodes=10, batch_size=32, save_model_name=None, save_evolution=True, evolution_interval=1):
    """
    Train an RL agent on the given data
    
    Parameters:
    -----------
    data: pandas.DataFrame
        Market data with OHLCV and indicators
    episodes: int
        Number of episodes to train for
    batch_size: int
        Batch size for replay
    save_model_name: str, optional
        Name to save the model under (if None, model is not saved)
    save_evolution: bool, optional
        Whether to save intermediate models to show evolution
    evolution_interval: int, optional
        Number of episodes between evolution checkpoints
        
    Returns:
    --------
    tuple
        (agent, training_results)
    """
    # Create environment and agent
    env = TradingEnvironment(data)
    state_size = 6  # Number of features in state representation
    action_size = 3  # Hold, Buy, Sell
    agent = TradingRLAgent(state_size=state_size, action_size=action_size)
    
    # Training results
    results = {
        'episode_rewards': [],
        'portfolio_values': [],
        'trade_counts': [],
        'win_rates': [],
        'avg_profit_per_trade': [],
        'exploration_rates': [],
        'evolution_models': []  # Store paths to evolution models
    }
    
    # Training loop
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Episode loop
        trades_this_episode = []
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Accumulate reward
            episode_reward += reward
            
            # Track trade if completed
            if 'trade_completed' in info and info['trade_completed']:
                trades_this_episode.append(info['trade'])
        
        # Train agent
        agent.replay(batch_size)
        
        # Calculate additional metrics for this episode
        if trades_this_episode:
            # Win rate
            profits = [trade.get('profit_pct', 0) for trade in trades_this_episode]
            wins = sum(p > 0 for p in profits)
            win_rate = wins / len(trades_this_episode) if trades_this_episode else 0
            
            # Average profit per trade
            avg_profit = sum(profits) / len(trades_this_episode) if trades_this_episode else 0
        else:
            win_rate = 0
            avg_profit = 0
        
        # Record results
        results['episode_rewards'].append(episode_reward)
        results['portfolio_values'].append(info['portfolio_value'])
        results['trade_counts'].append(len(env.trades))
        results['win_rates'].append(win_rate)
        results['avg_profit_per_trade'].append(avg_profit)
        results['exploration_rates'].append(agent.epsilon)
        
        # Save evolution model if requested
        if save_evolution and (episode % evolution_interval == 0 or episode == episodes - 1):
            if save_model_name:
                evolution_model_name = f"{save_model_name}_evolution_{episode+1}"
                model_path = agent.save_model(evolution_model_name)
                results['evolution_models'].append({
                    'episode': episode + 1,
                    'model_path': model_path,
                    'epsilon': agent.epsilon,
                    'reward': episode_reward,
                    'portfolio_value': info['portfolio_value'],
                    'trades': len(env.trades),
                    'win_rate': win_rate,
                    'avg_profit': avg_profit
                })
        
        # Print progress
        print(f"Episode: {episode+1}/{episodes}, Reward: {episode_reward:.2f}, "
              f"Portfolio Value: {info['portfolio_value']:.2f}, "
              f"Trades: {len(env.trades)}, Win Rate: {win_rate:.2f}")
    
    # Save final model if requested
    if save_model_name:
        agent.save_model(save_model_name)
    
    return agent, results


def evaluate_agent(agent, data):
    """
    Evaluate a trained agent on the given data
    
    Parameters:
    -----------
    agent: TradingRLAgent
        Trained agent
    data: pandas.DataFrame
        Market data to evaluate on
        
    Returns:
    --------
    dict
        Evaluation results
    """
    # Create environment
    env = TradingEnvironment(data)
    
    # Reset environment
    state = env.reset()
    done = False
    
    # Turn off exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0
    
    # Episode loop
    while not done:
        # Choose action
        action = agent.act(state)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Update state
        state = next_state
    
    # Restore exploration rate
    agent.epsilon = old_epsilon
    
    # Calculate results
    initial_value = env.initial_balance
    final_value = info['portfolio_value']
    profit = final_value - initial_value
    roi = (profit / initial_value) * 100
    
    # Benchmark: Buy and Hold
    first_price = data.iloc[0]['Close']
    last_price = data.iloc[-1]['Close']
    shares_bought = initial_value / first_price
    hold_value = shares_bought * last_price
    hold_profit = hold_value - initial_value
    hold_roi = (hold_profit / initial_value) * 100
    
    # Results
    results = {
        'initial_value': initial_value,
        'final_value': final_value,
        'profit': profit,
        'roi': roi,
        'trades': len(env.trades),
        'hold_value': hold_value,
        'hold_profit': hold_profit,
        'hold_roi': hold_roi,
        'trades_list': env.trades
    }
    
    return results


def get_all_models():
    """
    Get a list of all saved models
    
    Returns:
    --------
    list
        List of model filenames
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    models = []
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.json'):
            models.append(filename)
    
    return models


# Main function for testing
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31')
    n = len(dates)
    
    # Simulate a price series
    close = 100 + np.cumsum(np.random.normal(0, 1, n))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': close * np.random.normal(0.999, 0.001, n),
        'High': close * np.random.normal(1.01, 0.005, n),
        'Low': close * np.random.normal(0.99, 0.005, n),
        'Close': close,
        'Volume': np.random.normal(1000000, 200000, n)
    }, index=dates)
    
    # Add indicators
    data['Returns'] = data['Close'].pct_change()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values
    data = data.dropna()
    
    # Train agent
    print("Training agent...")
    agent, results = train_agent(data, episodes=5, batch_size=32, save_model_name="test_model")
    
    # Evaluate agent
    print("\nEvaluating agent...")
    eval_results = evaluate_agent(agent, data)
    
    # Print results
    print(f"\nInitial Value: ${eval_results['initial_value']:.2f}")
    print(f"Final Value: ${eval_results['final_value']:.2f}")
    print(f"Profit: ${eval_results['profit']:.2f} ({eval_results['roi']:.2f}%)")
    print(f"Number of Trades: {eval_results['trades']}")
    print(f"\nBuy and Hold Value: ${eval_results['hold_value']:.2f}")
    print(f"Buy and Hold Profit: ${eval_results['hold_profit']:.2f} ({eval_results['hold_roi']:.2f}%)")
    
    print("\nAgent vs Buy and Hold:")
    if eval_results['roi'] > eval_results['hold_roi']:
        print(f"Agent outperformed Buy and Hold by {eval_results['roi'] - eval_results['hold_roi']:.2f}%")
    else:
        print(f"Buy and Hold outperformed Agent by {eval_results['hold_roi'] - eval_results['roi']:.2f}%")