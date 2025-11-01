"""RL agent for optimizing image generation prompts based on virality."""
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from collections import deque

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config
from src.observability.metrics import MetricsLogger

class PromptOptimizer(nn.Module):
    """Neural network for prompt optimization."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        """Initialize prompt optimizer network."""
        super(PromptOptimizer, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class RLPromptOptimizer:
    """RL agent that learns to optimize prompts based on virality rewards."""
    
    def __init__(self, use_wandb: bool = True):
        """Initialize RL prompt optimizer."""
        self.algorithm = config.get('rl.algorithm', 'PPO')
        
        # Convert config values to proper types
        lr = config.get('rl.learning_rate', 3e-4)
        self.learning_rate = float(lr) if isinstance(lr, (int, float, str)) else 3e-4
        
        bs = config.get('rl.batch_size', 32)
        self.batch_size = int(bs) if isinstance(bs, (int, float, str)) else 32
        
        uf = config.get('rl.update_frequency', 10)
        self.update_frequency = int(uf) if isinstance(uf, (int, float, str)) else 10
        
        rw = config.get('rl.reward_weight', 1.0)
        self.reward_weight = float(rw) if isinstance(rw, (int, float, str)) else 1.0
        
        er = config.get('rl.exploration_rate', 0.1)
        self.exploration_rate = float(er) if isinstance(er, (int, float, str)) else 0.1
        self.use_wandb = use_wandb
        self.metrics = MetricsLogger() if use_wandb else None
        
        # RL state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PromptOptimizer().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Experience buffer
        self.experience_buffer: deque = deque(maxlen=1000)
        
        # Statistics
        self.step_count = 0
        self.total_rewards = []
    
    def optimize_prompt(
        self, 
        base_prompt: str, 
        context: Dict[str, Any],
        virality_score: float
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize a prompt based on context and virality feedback.
        
        Args:
            base_prompt: Original prompt
            context: Video context
            virality_score: Current virality score
            
        Returns:
            Tuple of (optimized_prompt, optimization_metadata)
        """
        try:
            # Convert prompt and context to feature vector
            state = self._encode_state(base_prompt, context, virality_score)
            
            # Get action (prompt modifications)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Apply modifications to prompt
            optimized_prompt = self._apply_modifications(base_prompt, action)
            
            # Store experience (will be updated with reward later)
            self.experience_buffer.append({
                'state': state,
                'action': action,
                'base_prompt': base_prompt,
                'optimized_prompt': optimized_prompt,
                'virality_score': virality_score,
                'reward': None  # Will be set when we get feedback
            })
            
            metadata = {
                'optimization_applied': True,
                'action_magnitude': np.linalg.norm(action),
                'exploration_rate': self.exploration_rate,
                'step_count': self.step_count
            }
            
            self.step_count += 1
            
            # Log metrics
            if self.metrics:
                self.metrics.log_rl_metrics({
                    'action_magnitude': np.linalg.norm(action),
                    'virality_score': virality_score,
                    'step_count': self.step_count,
                    'buffer_size': len(self.experience_buffer)
                })
            
            return optimized_prompt, metadata
        
        except Exception as e:
            error_msg = f"Prompt optimization failed: {str(e)}"
            
            if self.metrics:
                self.metrics.log_error('rl_optimization', error_msg)
            
            return base_prompt, {'optimization_applied': False, 'error': error_msg}
    
    def update_policy(self, reward: float, virality_score: float):
        """
        Update RL policy based on reward.
        
        Args:
            reward: Reward signal (typically virality score improvement)
            virality_score: Current virality score
        """
        if len(self.experience_buffer) < self.batch_size:
            return
        
        try:
            # Update last experience with reward
            if self.experience_buffer:
                self.experience_buffer[-1]['reward'] = reward
            
            # Sample batch
            batch = list(self.experience_buffer)[-self.batch_size:]
            
            # Prepare batch data
            states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
            actions = torch.FloatTensor([exp['action'] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] or 0.0 for exp in batch]).to(self.device)
            
            # Normalize rewards
            if rewards.std() > 0:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Policy gradient update (simplified)
            predicted_actions = self.policy_net(states)
            
            # Compute loss (encourage actions that lead to higher rewards)
            loss = -torch.mean(predicted_actions * rewards.unsqueeze(1))
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Log metrics
            if self.metrics:
                self.metrics.log_rl_update_metrics({
                    'loss': loss.item(),
                    'mean_reward': rewards.mean().item(),
                    'std_reward': rewards.std().item() if len(rewards) > 1 else 0.0,
                    'virality_score': virality_score
                })
            
            self.total_rewards.append(reward)
        
        except Exception as e:
            error_msg = f"Policy update failed: {str(e)}"
            
            if self.metrics:
                self.metrics.log_error('rl_update', error_msg)
    
    def _encode_state(
        self, 
        prompt: str, 
        context: Dict[str, Any],
        virality_score: float
    ) -> np.ndarray:
        """Encode state as feature vector."""
        # Simple encoding: concatenate prompt features, context features, and virality
        prompt_features = self._encode_text(prompt)
        context_features = self._encode_context(context)
        virality_feature = np.array([virality_score])
        
        # Combine features (pad/truncate to fixed size)
        features = np.concatenate([
            prompt_features[:256],  # First 256 dims
            context_features[:128],  # Next 128 dims
            virality_feature,  # 1 dim
            np.zeros(max(0, 512 - 256 - 128 - 1))  # Pad to 512
        ])
        
        return features[:512]
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Simple text encoding (can be improved with embeddings)."""
        # Use character-level frequency encoding
        text_lower = text.lower()
        features = np.zeros(256)
        
        for char in text_lower:
            if ord(char) < 256:
                features[ord(char)] += 1
        
        # Normalize
        if features.sum() > 0:
            features = features / features.sum()
        
        return features
    
    def _encode_context(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode context as feature vector."""
        # Extract key context values
        features = []
        
        # Vibe encoding (simple categorical)
        vibe = context.get('vibe', 'neutral').lower()
        vibe_map = {'energetic': 1.0, 'calm': 0.3, 'professional': 0.7, 'casual': 0.5}
        features.append(vibe_map.get(vibe, 0.5))
        
        # Color palette (count)
        colors = context.get('color_palette', [])
        features.append(len(colors))
        
        # Visual elements (count)
        elements = context.get('visual_elements', [])
        features.append(len(elements))
        
        # Pad to 128 dims
        features = np.array(features + [0.0] * (128 - len(features)))
        return features[:128]
    
    def _apply_modifications(self, base_prompt: str, action: np.ndarray) -> str:
        """Apply RL action modifications to prompt."""
        # Simple modification strategy: add style modifiers based on action
        modifications = []
        
        # Style modifiers based on action values
        if action[0] > 0.5:
            modifications.append("high quality, detailed")
        if action[1] > 0.5:
            modifications.append("cinematic, professional")
        if action[2] > 0.5:
            modifications.append("vibrant, colorful")
        if action[3] > 0.5:
            modifications.append("dynamic, energetic")
        
        if modifications:
            optimized_prompt = f"{base_prompt}, {', '.join(modifications)}"
        else:
            optimized_prompt = base_prompt
        
        return optimized_prompt

