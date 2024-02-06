import hydra
import omegaconf
from omegaconf import ValueNode
import torch
import pandas as pd



from torch.utils.data import Dataset
from torch_geometric.data import Data

class CrystalDataset(Dataset):
    def __init__(self, name: ValueNode, path:ValueNode,
                 prop:ValueNode, niggli:ValueNode, primitive:ValueNode,
                 graph_method:ValueNode, preprocess_workers:ValueNode,
                 lattice_scale_method:ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method


        self.cached_data = preprocess(
            self.path,
            preprocess
        )