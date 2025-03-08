import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ZIG(nn.Module):
    def __init__(self, yDim, xDim, gen_nodes, factor):
        """
        Args:
            yDim (int): Number of output dimensions.
            xDim (int): Number of input dimensions.
            gen_nodes (int): Number of hidden units in the hidden layers.
            factor (array-like or torch.Tensor): Constant used for 'loc'. 
                Should have shape (yDim,).
        """
        super(ZIG, self).__init__()
        
        # Define the layers:
        self.fc1 = nn.Linear(xDim, gen_nodes)
        self.fc2 = nn.Linear(gen_nodes, gen_nodes)
        self.fc_theta = nn.Linear(gen_nodes, yDim)
        self.fc_p = nn.Linear(gen_nodes, yDim)
        
        # Initialize weights with uniform distribution:
        rangeRate1 = 1.0 / math.sqrt(xDim)
        rangeRate2 = 1.0 / math.sqrt(gen_nodes)
        nn.init.uniform_(self.fc1.weight, -rangeRate1, rangeRate1)
        nn.init.uniform_(self.fc2.weight, -rangeRate2, rangeRate2)
        nn.init.uniform_(self.fc_theta.weight, -rangeRate2, rangeRate2)
        nn.init.uniform_(self.fc_p.weight, -rangeRate2, rangeRate2)
        
        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_theta.bias)
        nn.init.zeros_(self.fc_p.bias)
        
        # Create learnable parameter logk (initialized as zeros)
        self.logk = nn.Parameter(torch.zeros(yDim))
        
        # 'loc' comes from factor. Register it as a buffer so that it is not updated by the optimizer.
        if not torch.is_tensor(factor):
            factor = torch.tensor(factor, dtype=torch.float32)
        self.register_buffer('loc', factor)
        
    def forward(self, X, Y=None):
        # Pass input through the network with tanh activations:
        full1 = torch.tanh(self.fc1(X))
        full2 = torch.tanh(self.fc2(full1))
        full_theta = self.fc_theta(full2)
        full_p = self.fc_p(full2)
        
        # Compute predictions:
        theta = torch.exp(full_theta)
        p = torch.sigmoid(full_p)  # equivalent to exp(full_p)/(1+exp(full_p))
        
        # Compute k (learnable) and get loc (constant)
        k = torch.exp(self.logk) + 1e-7  # shape: (yDim,)
        
        # Compute rate with proper broadcasting:
        rate = (theta * k.unsqueeze(0) + self.loc.unsqueeze(0)) * p
        
        # If no target is provided, return predictions:
        if Y is None:
            return theta, k, p, self.loc, rate
        
        # Otherwise, compute the entropy loss:
        Nsamps = Y.shape[0]
        # Create a mask of non-zero elements in Y:
        mask = (Y != 0)
        
        # Expand k and loc to match Y's shape (Nsamps, yDim)
        k_NTxD = k.unsqueeze(0).expand(Nsamps, -1)
        loc_NTxD = self.loc.unsqueeze(0).expand(Nsamps, -1)
        
        # Select the nonzero entries:
        y_temp = Y[mask]
        r_temp = theta[mask]
        p_temp = p[mask]
        k_temp = k_NTxD[mask]
        loc_temp = loc_NTxD[mask]
        
        # Adjust for numerical stability:
        eps = 1e-6
        p_temp = p_temp * (1 - 2e-6) + 1e-6
        r_temp = r_temp + eps
        # Clamp the difference (y_temp - loc_temp) to avoid log(0) or log(negative)
        delta = torch.clamp(y_temp - loc_temp, min=eps)
        
        #LY1 = torch.sum(torch.log(p_temp) - k_temp * torch.log(r_temp) - (y_temp - loc_temp) / r_temp)
        #LY2 = torch.sum(-torch.lgamma(k_temp) + (k_temp - 1) * torch.log(delta))
        
        # For entries where Y == 0:
        gr_temp = p[~mask]
        LY3 = torch.sum(torch.log(1 - gr_temp + eps))  # add eps for safety

        gamma = 2  # Tunable hyperparameter, 2 is a common choice
        LY1_focal = torch.sum((1 - p_temp) ** gamma * (torch.log(p_temp) - k_temp * torch.log(r_temp) - (y_temp - loc_temp) / r_temp))
        LY2_focal = torch.sum((1 - p_temp) ** gamma * (-torch.lgamma(k_temp) + (k_temp - 1) * torch.log(delta)))

        entropy_loss = LY1_focal + LY2_focal + LY3
        
        #entropy_loss = LY1 + LY2 + LY3
        
        return entropy_loss, theta, k, p, self.loc, rate
