import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def gce_loss(outputs, Y, q):
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1-(pow_outputs*Y).sum(dim=1))/q 
    return sample_loss