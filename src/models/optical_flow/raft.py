from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights

def get_raft(small=False):
    if small:
        model = raft_small(weights = Raft_Small_Weights.DEFAULT)
    else:
        model = raft_large(weights = Raft_Large_Weights.C_T_V2)
    return model