'''

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

        self.dropout1 = nn.Dropout(p=0.9)
        self.dropout2 = nn.Dropout(p=0.9)
        
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
        full1 = self.dropout1(full1)
        full2 = torch.tanh(self.fc2(full1))
        full2 = self.dropout2(full2)
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

        gamma = 10  # Tunable hyperparameter, 2 is a common choice
        LY1_focal = torch.sum((1 - p_temp) ** gamma * (torch.log(p_temp) - k_temp * torch.log(r_temp) - (y_temp - loc_temp) / r_temp))
        LY2_focal = torch.sum((1 - p_temp) ** gamma * (-torch.lgamma(k_temp) + (k_temp - 1) * torch.log(delta)))

        entropy_loss = LY1_focal + LY2_focal + LY3
        
        #entropy_loss = LY1 + LY2 + LY3
        
        return entropy_loss, theta, k, p, self.loc, rate
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ZIG(nn.Module):
    def __init__(self, yDim, xDim, gen_nodes, alpha=0.25, gamma=2.0, p_dropout=0.9):
        """
        Predicts only whether a neuron has an event (Y>0), ignoring its magnitude.

        Args:
            yDim (int): Number of output dimensions (e.g. number of neurons).
            xDim (int): Number of input dimensions (e.g. embedding size).
            gen_nodes (int): Number of hidden units in the hidden layers.
            alpha (float): Weighting factor for positive examples in focal loss.
            gamma (float): Exponent for down-weighting easy examples in focal loss.
            p_dropout (float): Dropout probability.
        """
        super(ZIG, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(xDim, gen_nodes)
        self.fc2 = nn.Linear(gen_nodes, gen_nodes)
        self.fc_p = nn.Linear(gen_nodes, yDim)

        # Optional dropout layers to help with regularization
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.dropout2 = nn.Dropout(p=p_dropout)

        # Initialize weights
        rangeRate1 = 1.0 / math.sqrt(xDim)
        rangeRate2 = 1.0 / math.sqrt(gen_nodes)
        nn.init.uniform_(self.fc1.weight, -rangeRate1, rangeRate1)
        nn.init.uniform_(self.fc2.weight, -rangeRate2, rangeRate2)
        nn.init.uniform_(self.fc_p.weight, -rangeRate2, rangeRate2)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_p.bias)

        # Store alpha and gamma for focal loss
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, X, Y=None):
        """
        Forward pass. If Y is given, returns the focal loss. Otherwise, returns the predicted probabilities p.

        Args:
            X (torch.Tensor): Input features of shape (N, xDim).
            Y (torch.Tensor, optional): Target data of shape (N, yDim). 
                - We consider an event to have occurred if Y>0.

        Returns:
            If Y is None:
                p (torch.Tensor): Probability of event shape (N, yDim)
            If Y is not None:
                focal_loss (torch.Tensor): Scalar loss.
                p (torch.Tensor): Probability of event shape (N, yDim)
        """
        # Pass input through the network
        full1 = torch.tanh(self.fc1(X))
        full1 = self.dropout1(full1)
        full2 = torch.tanh(self.fc2(full1))
        full2 = self.dropout2(full2)

        # This is our logit for the probability of an event
        logits_p = self.fc_p(full2)
        # Convert to probability
        p = torch.sigmoid(logits_p)

        if Y is None:
            # If no labels are provided, just return probabilities
            return p

        # Otherwise, compute the focal loss
        # 1) Convert Y>0 to a binary event label
        event_label = (Y > 0).float()

        # 2) Compute the standard binary cross-entropy for each output
        # We'll do this in a 'none' reduction so we can apply focal weighting manually
        bce_loss = F.binary_cross_entropy_with_logits(
            logits_p, event_label, reduction='none'
        )
        
        # 3) From the link on focal loss, we define:
        #       pt = p if y=1 else (1-p)
        # We can get pt by doing:
        pt = event_label * p + (1 - event_label) * (1 - p)

        # 4) alpha weighting for positive vs negative labels
        #    If an element is positive => alpha_factor = alpha
        #       else => alpha_factor = 1 - alpha
        alpha_factor = event_label * self.alpha + (1 - event_label) * (1 - self.alpha)

        # 5) Focal loss scaling
        #    focal_weight = alpha_factor * (1 - pt)^gamma
        focal_weight = alpha_factor * ((1 - pt) ** self.gamma)

        # 6) Combine everything: focal loss = focal_weight * BCE
        focal_loss_elementwise = focal_weight * bce_loss

        # Sum (or mean) over all elements. Here, we'll just sum.
        # You could also do a .mean() if you prefer an average loss.
        focal_loss = focal_loss_elementwise.sum()

        return focal_loss, p
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ZIG(nn.Module):
    def __init__(self,xDim, neuronDim):
        super(ZIG, self).__init__()

        self.fc1= nn.Linear(xDim, neuronDim)
        
    def forward(self, X, Y=None):
        # Pass input through the network
        logits_p= self.fc1(X)
        p= torch.sigmoid(logits_p)
        if Y is None:
            # If no labels are provided, just return probabilities
            return p

        # Otherwise, compute the focal loss
        # 1) Convert Y>0 to a binary event label
        event_label = (Y > 0).float()

        # 2) Compute the standard binary cross-entropy for each output
        # We'll do this in a 'none' reduction so we can apply focal weighting manually

        p = torch.sigmoid(logits_p)
        event_label = (Y > 0).float()

        # Focal loss terms
        alpha = 0.05
        gamma = 2.0
        bce = F.binary_cross_entropy_with_logits(logits_p, event_label, reduction='none')

        pt = torch.exp(-bce)
        focal_weight = alpha * (1 - pt) ** gamma
        focal_loss = focal_weight * bce

        return focal_loss, p

