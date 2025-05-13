import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
import time

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for RL model training
    """
    
    def __init__(self, data, features, action_type='discrete', reward_config=None):
        """
        Initialize the trading environment
        
        Parameters:
        -----------
        data: pandas.DataFrame
            DataFrame containing market data with technical indicators
        features: list
            List of column names to use as features
        action_type: str
            Type of action space ('discrete' or 'continuous')
        reward_config: dict
            Configuration for the reward function
        """
        super(TradingEnv, self).__init__()
        
        # Store environment parameters
        self.data = data.copy()
        self.features = features
        self.action_type = action_type
        self.reward_config = reward_config if reward_config else {"type": "return"}
        
        # Initialize state
        self.current_step = 0
        self.total_steps = len(self.data) - 1
        self.current_position = 0  # 0: no position, 1: long, -1: short
        self.portfolio_value = 1.0
        self.initial_portfolio_value = 1.0
        self.returns = []
        
        # Normalize features (important for neural network inputs)
        self._normalize_features()
        
        # Define action space
        if action_type == 'discrete':
            # 0: sell, 1: hold, 2: buy
            self.action_space = spaces.Discrete(3)
        else:
            # Continuous action space for position sizing (-1.0 to 1.0)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Define observation space
        # +3 for current position, portfolio value, and days held
        feature_dimension = len(features) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dimension,), dtype=np.float32
        )
        
        # Initialize additional variables
        self.current_trade_start = None
        self.days_in_position = 0
        self.trade_returns = []
        self.max_portfolio_value = 1.0
    
    def _normalize_features(self):
        """
        Normalize features to have mean 0 and standard deviation 1
        """
        self.feature_means = {}
        self.feature_stds = {}
        
        for feature in self.features:
            if feature in self.data.columns:
                self.feature_means[feature] = self.data[feature].mean()
                self.feature_stds[feature] = self.data[feature].std()
                
                # Replace zeros in std to avoid division by zero
                if self.feature_stds[feature] == 0:
                    self.feature_stds[feature] = 1.0
    
    def reset(self, seed=None):
        """
        Reset the environment to initial state
        
        Returns:
        --------
        observation: numpy.ndarray
            The initial observation
        info: dict
            Additional information
        """
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.current_position = 0
        self.portfolio_value = 1.0
        self.initial_portfolio_value = 1.0
        self.returns = []
        self.current_trade_start = None
        self.days_in_position = 0
        self.trade_returns = []
        self.max_portfolio_value = 1.0
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        """
        Take a step in the environment
        
        Parameters:
        -----------
        action: int or float
            The action to take
            
        Returns:
        --------
        observation: numpy.ndarray
            The observation after the step
        reward: float
            The reward for the step
        terminated: bool
            Whether the episode is terminated
        truncated: bool
            Whether the episode is truncated
        info: dict
            Additional information
        """
        # Get current values
        current_price = self.data['Close'].iloc[self.current_step]
        
        # Calculate position based on action
        if self.action_type == 'discrete':
            # 0: sell, 1: hold, 2: buy
            new_position = -1 if action == 0 else (1 if action == 2 else 0)
        else:
            # Continuous action space (-1.0 to 1.0)
            new_position = float(action[0])
        
        # Execute the trade
        if self.current_step < self.total_steps:
            # Get next price
            next_price = self.data['Close'].iloc[self.current_step + 1]
            
            # Calculate returns
            price_return = next_price / current_price - 1
            
            # Apply position to returns
            position_for_return = self.current_position if new_position == self.current_position else new_position
            strategy_return = price_return * position_for_return
            
            # Update portfolio value
            self.portfolio_value *= (1 + strategy_return)
            self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
            
            # Store returns for calculating metrics
            self.returns.append(strategy_return)
            
            # Update position tracking
            if new_position != 0:
                if self.current_position == 0:  # Starting a new position
                    self.current_trade_start = self.current_step
                    self.days_in_position = 1
                else:  # Continuing or changing position
                    self.days_in_position += 1
            else:
                if self.current_position != 0:  # Closing a position
                    # Calculate trade return
                    trade_start_price = self.data['Close'].iloc[self.current_trade_start]
                    trade_end_price = next_price
                    trade_return = (trade_end_price / trade_start_price - 1) * self.current_position
                    self.trade_returns.append(trade_return)
                    
                self.days_in_position = 0
            
            # Update position
            self.current_position = new_position
            
            # Increment step
            self.current_step += 1
            
            # Get reward
            reward = self._calculate_reward(strategy_return)
            
            # Get new observation
            observation = self._get_observation()
            
            # Check if episode is done
            done = self.current_step >= self.total_steps
            
            # Information for monitoring
            info = {
                'portfolio_value': self.portfolio_value,
                'return': strategy_return,
                'position': self.current_position
            }
            
            return observation, reward, done, False, info
        else:
            # If we've reached the end of the data
            observation = self._get_observation()
            return observation, 0, True, False, {'portfolio_value': self.portfolio_value}
    
    def _get_observation(self):
        """
        Get the current observation (state)
        
        Returns:
        --------
        numpy.ndarray
            The observation vector
        """
        # Get feature values
        feature_values = []
        
        for feature in self.features:
            if feature in self.data.columns:
                # Get raw value
                value = self.data[feature].iloc[self.current_step]
                
                # Normalize value
                norm_value = (value - self.feature_means[feature]) / self.feature_stds[feature]
                feature_values.append(norm_value)
        
        # Add current position
        feature_values.append(self.current_position)
        
        # Add portfolio value relative to initial
        feature_values.append(self.portfolio_value / self.initial_portfolio_value - 1)
        
        # Add days in current position
        feature_values.append(self.days_in_position / 10)  # Normalize days
        
        # Convert to numpy array
        observation = np.array(feature_values, dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, current_return):
        """
        Calculate reward based on configured reward function
        
        Parameters:
        -----------
        current_return: float
            The return for the current step
            
        Returns:
        --------
        float
            The calculated reward
        """
        reward_type = self.reward_config.get("type", "return")
        
        if reward_type == "return":
            # Simple return
            reward = current_return
        
        elif reward_type == "return_risk":
            # Return with risk penalty
            risk_factor = self.reward_config.get("risk_factor", 0.5)
            
            # Calculate risk penalty
            if len(self.returns) > 1:
                volatility = np.std(self.returns[-20:]) if len(self.returns) >= 20 else np.std(self.returns)
                drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
                risk_penalty = volatility * drawdown * risk_factor
            else:
                risk_penalty = 0
            
            reward = current_return - risk_penalty
        
        elif reward_type == "sharpe":
            # Sharpe ratio based reward
            if len(self.returns) > 1:
                returns_mean = np.mean(self.returns[-20:]) if len(self.returns) >= 20 else np.mean(self.returns)
                returns_std = np.std(self.returns[-20:]) if len(self.returns) >= 20 else np.std(self.returns)
                sharpe = returns_mean / returns_std if returns_std > 0 else 0
                
                # Scale sharpe to make it a more appropriate reward
                reward = sharpe * 0.1 + current_return
            else:
                reward = current_return
        
        elif reward_type == "custom":
            # Custom reward function with configurable weights
            return_weight = self.reward_config.get("return_weight", 0.7)
            risk_weight = self.reward_config.get("risk_weight", 0.3)
            consistency_weight = self.reward_config.get("consistency_weight", 0.2)
            
            # Return component
            return_component = current_return * return_weight
            
            # Risk component
            risk_component = 0
            if len(self.returns) > 1:
                drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
                risk_component = -drawdown * risk_weight
            
            # Consistency component
            consistency_component = 0
            if len(self.trade_returns) > 1:
                win_rate = sum(1 for r in self.trade_returns if r > 0) / len(self.trade_returns)
                consistency_component = (win_rate - 0.5) * consistency_weight
            
            reward = return_component + risk_component + consistency_component
        
        else:
            # Default to simple return
            reward = current_return
        
        return reward


class TrainingCallback(BaseCallback):
    """
    Callback for monitoring training progress
    """
    
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.rewards = []
        self.portfolio_values = []
    
    def _on_step(self):
        # Record rewards and portfolio values
        self.rewards.append(self.locals.get('rewards', [0])[0])
        info = self.locals.get('infos', [{}])[0]
        self.portfolio_values.append(info.get('portfolio_value', 1.0))
        return True


class RLModel:
    """
    Class for creating and training RL models for trading
    """
    
    def __init__(self, data, features, action_type='discrete', model_type='ppo', learning_rate=1e-4, gamma=0.99, reward_config=None):
        """
        Initialize the RL model
        
        Parameters:
        -----------
        data: pandas.DataFrame
            DataFrame containing market data with technical indicators
        features: list
            List of column names to use as features
        action_type: str
            Type of action space ('discrete' or 'continuous')
        model_type: str
            Type of RL model ('a2c', 'ppo', or 'dqn')
        learning_rate: float
            Learning rate for the model
        gamma: float
            Discount factor for future rewards
        reward_config: dict
            Configuration for the reward function
        """
        self.data = data.copy()
        self.features = features
        self.action_type = action_type
        self.model_type = model_type.lower()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reward_config = reward_config if reward_config else {"type": "return"}
        
        # Create environment
        self.env = TradingEnv(
            data=self.data,
            features=self.features,
            action_type=self.action_type,
            reward_config=self.reward_config
        )
        
        # Initialize model
        self.model = None
        self._create_model()
    
    def _create_model(self):
        """
        Create the RL model based on the specified type
        """
        if self.model_type == 'a2c':
            self.model = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=0
            )
        elif self.model_type == 'ppo':
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=0
            )
        elif self.model_type == 'dqn':
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, episodes=1000, validation_split=0.2):
        """
        Train the RL model
        
        Parameters:
        -----------
        episodes: int
            Number of episodes to train for
        validation_split: float
            Fraction of data to use for validation
            
        Returns:
        --------
        dict
            Dictionary with training results
        """
        # Split data into training and validation sets
        train_size = int(len(self.data) * (1 - validation_split))
        train_data = self.data.iloc[:train_size].copy()
        val_data = self.data.iloc[train_size:].copy()
        
        # Create training environment
        train_env = TradingEnv(
            data=train_data,
            features=self.features,
            action_type=self.action_type,
            reward_config=self.reward_config
        )
        
        # Create callback for monitoring
        callback = TrainingCallback()
        
        # Train the model
        start_time = time.time()
        self.model.learn(total_timesteps=episodes, callback=callback)
        training_time = time.time() - start_time
        
        # Evaluate on validation set
        val_env = TradingEnv(
            data=val_data,
            features=self.features,
            action_type=self.action_type,
            reward_config=self.reward_config
        )
        
        # Run model on validation data
        val_results = self._evaluate(val_env)
        
        # Training results
        training_results = {
            'episode_rewards': callback.rewards,
            'final_reward': callback.rewards[-1] if callback.rewards else 0,
            'mean_reward': np.mean(callback.rewards) if callback.rewards else 0,
            'max_reward': np.max(callback.rewards) if callback.rewards else 0,
            'training_time': training_time,
            'validation_equity_curve': val_results['equity_curve'],
            'model_actions': val_results['actions']
        }
        
        return training_results
    
    def _evaluate(self, env):
        """
        Evaluate the model on a given environment
        
        Parameters:
        -----------
        env: TradingEnv
            Environment to evaluate on
            
        Returns:
        --------
        dict
            Dictionary with evaluation results
        """
        # Reset environment
        obs, _ = env.reset()
        
        done = False
        total_reward = 0
        portfolio_values = [1.0]
        actions = []
        buy_signals = []
        sell_signals = []
        current_position = 0
        
        while not done:
            # Predict action
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record results
            total_reward += reward
            portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
            
            # Track position changes for buy/sell signals
            new_position = info.get('position', 0)
            if new_position > current_position:
                buy_signals.append(env.current_step)
            elif new_position < current_position:
                sell_signals.append(env.current_step)
            
            current_position = new_position
            actions.append(action)
        
        # Create result dictionary
        result = {
            'total_reward': total_reward,
            'final_portfolio_value': portfolio_values[-1],
            'equity_curve': pd.Series(portfolio_values),
            'actions': {
                'raw': actions,
                'buy': buy_signals,
                'sell': sell_signals
            }
        }
        
        return result
    
    def predict(self, data=None):
        """
        Make predictions with the trained model
        
        Parameters:
        -----------
        data: pandas.DataFrame
            Data to make predictions on (defaults to the original data)
            
        Returns:
        --------
        dict
            Dictionary with prediction results
        """
        if data is None:
            data = self.data
        
        # Create environment for prediction
        pred_env = TradingEnv(
            data=data,
            features=self.features,
            action_type=self.action_type,
            reward_config=self.reward_config
        )
        
        # Run prediction
        pred_results = self._evaluate(pred_env)
        
        return pred_results
