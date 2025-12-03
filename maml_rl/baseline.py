import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearFeatureBaseline(nn.Module):
    """
    Linear baseline based on handcrafted features.
    Features: [observations, observations^2, time_step, time_step^2, time_step^3, ones]
    """
    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        
        # Feature size = 2 * input_size + 4 time features
        self.feature_size = 2 * input_size + 4
        
        # We use nn.Linear for simplicity and efficient parameter management
        self.linear = nn.Linear(self.feature_size, 1, bias=False)

    def _feature(self, episodes):
        # FIX 1: Handle both BatchEpisodes (object) and Tensor (raw observations)
        if torch.is_tensor(episodes):
            observations = episodes
            
            # Handle (Time, Dim) -> (1, Time, Dim)
            if observations.dim() == 2:
                observations = observations.unsqueeze(0)
            elif observations.dim() == 1:
                # Handle empty or 1D observations (e.g. from empty episodes)
                # FIX: Reshape using self.input_size to preserve feature dimension
                observations = observations.view(1, -1, self.input_size)
            
            batch_size, seq_len, _ = observations.shape
            # Default mask of ones for raw tensor input
            ones = torch.ones((batch_size, seq_len), device=observations.device)
        else:
            # BatchEpisodes object
            observations = episodes.observations
            ones = episodes.mask

        # Create time features
        # ones shape: (Batch, Time) -> (Batch, Time, 1)
        ones = ones.unsqueeze(2)
        
        batch_size, seq_len, _ = observations.shape
        
        # Time steps: normalized to [0, 1] roughly (div by 100)
        time_step = torch.arange(seq_len, device=observations.device).float()
        time_step = time_step.view(1, -1, 1).expand(batch_size, seq_len, 1)
        time_step = time_step * ones / 100.0

        # FIX 2: Added observations^2 to match original implementation features
        return torch.cat([
            observations,
            observations ** 2,
            time_step,
            time_step**2,
            time_step**3,
            ones
        ], dim=2)

    def fit(self, episodes):
        # sequence_length * batch_size x feature_size
        featmat = self._feature(episodes).view(-1, self.feature_size)
        # sequence_length * batch_size x 1
        returns = episodes.returns.view(-1, 1)

        # Remove blank (all-zero) episodes that only exist because episode lengths vary
        # (This masking logic was in your original code)
        if hasattr(episodes, 'mask'):
            flat_mask = episodes.mask.flatten()
            # Ensure mask is boolean or byte for indexing
            flat_mask = flat_mask > 0
            featmat = featmat[flat_mask]
            returns = returns[flat_mask]

        # Regularization (L2)
        reg_coeff = self._reg_coeff
        eye = torch.eye(self.feature_size, dtype=torch.float32,
                        device=self.linear.weight.device)
        
        # FIX 3: Modern PyTorch solver (torch.linalg.solve/lstsq)
        # Solve Normal Equations: (A^T A + lambda I) x = A^T b
        mat_a = torch.matmul(featmat.t(), featmat) + reg_coeff * eye
        mat_b = torch.matmul(featmat.t(), returns)
        
        try:
            coeffs = torch.linalg.solve(mat_a, mat_b)
        except RuntimeError:
            # Fallback for singular matrix
            coeffs = torch.linalg.lstsq(mat_a, mat_b).solution

        # Update linear layer weights
        self.linear.weight.data.copy_(coeffs.t())

    def forward(self, episodes):
        features = self._feature(episodes)
        return self.linear(features)