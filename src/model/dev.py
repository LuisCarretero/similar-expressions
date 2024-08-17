import os
import sys
import numpy as np
import torch


a = np.array([1.0, 2, 3])
print(a.dtype)

b = torch.tensor([1.0, 2, 3])
print(b.dtype)