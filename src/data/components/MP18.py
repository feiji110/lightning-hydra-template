
import warnings
warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF:")

import torch

import logging
import os.path as osp
import numpy as np
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset,Data
from utils.data_helpers import (
    clean_up,
    generate_edge_features,
    generate_node_features,
    get_cutoff_distance_matrix,
)
from torch_geometric.utils import dense_to_sparse
from torch_geometric.transforms import Compose
from utils.transforms import GetY
import torch_geometric.transforms as T
class MP18(InMemoryDataset):

    def __init__(self, root='data/', name='MP18', matbenchRaw=None, transform=None, pre_transform=[GetY()], r=8.0, n_neighbors=12, edge_steps=50, image_selfloop=True, points=100,target_name="formation_energy_per_atom"):
        
        self.name = name.lower()
        assert self.name in ['mp18', 'pt','2d','mof','surface','cubic', 'cif', 'matbench']
        self.r = r
        self.n_neighbors = n_neighbors
        self.edge_steps = edge_steps
        self.image_selfloop = image_selfloop
        self.points = points# dataset snumbers
        self.target_name = target_name # target property name
        self.device = torch.device('cpu')
        self._rawDf = matbenchRaw
        super(MP18, self).__init__(root, transform, pre_transform)
        if self.name == 'matbench':
            self.data, self.slices, self.matbench_train_test = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self):
        if self.name == 'cif':
            return ''
        else:
            return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        if self.name == 'mp18':
            file_names = ['mp.2018.6.1.json.zip']
        elif self.name == 'pt':
            file_names = ['pt.2023.5.19.json.zip']
        elif self.name == 'mof':
            file_names = ['mof.2023.5.19.json.zip']
        elif self.name == '2d':
            file_names = ['2d.2023.5.19.json.zip']
        elif self.name == 'surface':
            file_names = ['surface.2023.5.19.json.zip']
        elif self.name == 'cubic':
            file_names = ['cubic.2023.7.13.json.zip']
        elif self.name == 'cif':
            from glob import glob
            file_names = glob(f"{self.root}/*.cif")
        else:
            exit(1)
        return file_names

    @property
    def processed_file_names(self):
        processed_name = 'data_{}_{}_{}_{}_{}_{}_{}.pt'.format(self.name, self.r, self.n_neighbors, self.edge_steps, self.image_selfloop, self.points, self.target_name)
        return [processed_name]


    def process(self):
        logging.info("Data found at {}".format(self.raw_dir))
        # 1. get dict format
        dict_structures, y = self.json_wrap()

        # 2. # get Data list
        data_list = self.get_data_list(dict_structures, y) # 获取列表

        # 3. save 
        data, slices = self.collate(data_list)
        if self.name == 'matbench':
            torch.save((data, slices, self.matbench_train_test), self.processed_paths[0])
        else:
            torch.save((data, slices), self.processed_paths[0])
        logging.info("Processed data saved successfully.")
        
    def __str__(self):
        return '{}_{}_{}_{}_{}_{}_{}.pt'.format(self.name, self.r, self.n_neighbors, self.edge_steps, self.image_selfloop, self.points, self.target_name)
    
    def __repr__(self):
        return '{}()'.format(self.name)
    
    def pymatgen2ase(self,pymat_structure):
        from pymatgen.io.ase import AseAtomsAdaptor
        Adaptor = AseAtomsAdaptor()
        return Adaptor.get_atoms(pymat_structure)

    def json_wrap(self):
        import pandas as pd
        import os
        logging.info("Reading individual structures using Pymatgen.")

        from pymatgen.core import Structure
        if self.name.lower() in ['cif']:
            cifFiles = []
            for i in self.raw_paths:
                with open(i, 'r') as f:
                    strContent = f.read()
                cifFiles.append(strContent)
            ids = [os.path.basename(i).split('.')[0] for i in self.raw_paths]
            df = pd.DataFrame({'structure': cifFiles, 'material_id': ids, 'property': [.0]*len(ids)})
        elif self.name.lower() in ['matbench']:
            trainX, trainY = self._rawDf['trainX'], self._rawDf['trainY']
            testX, testY = self._rawDf['testX'], self._rawDf['testY']
            trainDf = pd.merge(trainX, trainY, on=trainX.index.name)
            testDf = pd.merge(testX, testY, on=testX.index.name)
            self.matbench_train_test = (len(trainDf), len(testDf))
            rawDf = pd.concat([trainDf, testDf])
            rawDf['material_id'] = rawDf.index
            df = rawDf
        else:
            df  = pd.read_json(self.raw_paths[0]) 
        logging.info("Converting data to standardized form(dict format) for downstream processing.")

        dict_structures = []
        for i, s in enumerate(tqdm(df["structure"])):
            if i == self.points:  # limit the dataset size
                break
            s = Structure.from_str(s, fmt="cif") if self.name != 'matbench' else s
            s = self.pymatgen2ase(s)
            d = {}
            pos = torch.tensor(s.get_positions(), dtype=torch.float)  
            cell = torch.tensor(
                np.array(s.get_cell()), dtype=torch.float
            ) # lattice vector 3*3 
            atomic_numbers = torch.LongTensor(s.get_atomic_numbers())

            if self.name == 'cubic':
                def getAB(element):
                    if df['A'][i] == element:
                        return 7
                    elif df['B'][i] == element:
                        return 8
                    else:
                        return 9
                d["AB"] = torch.LongTensor([getAB(i)  for i in s.get_chemical_symbols()])

            d["positions"] = pos
            d["cell"] = cell
            d["atomic_numbers"] = atomic_numbers
            d["structure_id"] = str(df['material_id'][i])


            _atoms_index     = s.get_atomic_numbers()
            # from utils.data_helpers import create_global_feat
            # gatgnn_glob_feat = create_global_feat(_atoms_index)
            # gatgnn_glob_feat = np.repeat(gatgnn_glob_feat,len(_atoms_index),axis=0) # 作用？
            # d["gatgnn_glob_feat"] = torch.Tensor(gatgnn_glob_feat).float()


            dict_structures.append(d)


            y = df[[self.target_name]].to_numpy()

            ##Compile structure sizes (# of atoms) and elemental compositions
            if i == 0:
                length = [len(_atoms_index)]
                elements = [list(set(_atoms_index))]
            else:
                length.append(len(_atoms_index))
                elements.append(list(set(_atoms_index)))
            n_atoms_max = max(length)
        species = list(set(sum(elements, [])))
        species.sort()
        num_species = len(species)
        print(
            "Max structure size: ", # Maximum number of unit cell atoms in the dataset
            n_atoms_max,
            "Max number of elements: ", # The number of distinct elements contained in the data set
            num_species,
        )
        return dict_structures, y
    
    def get_data_list(self, dict_structures, y):
        print(f"entering")
        n_structures = len(dict_structures)
        data_list = [Data() for _ in range(n_structures)]

        logging.info("Getting torch_geometric.data.Data() objects.")

        for i, sdict in enumerate(tqdm(dict_structures)):
            target_val = y[i]
            data = data_list[i]

            pos = sdict["positions"]
            cell = sdict["cell"]
            atomic_numbers = sdict["atomic_numbers"]
            structure_id = sdict["structure_id"]

            cd_matrix, cell_offsets = get_cutoff_distance_matrix(
                pos,
                cell,
                self.r,
                self.n_neighbors,
                image_selfloop=self.image_selfloop,
                device=self.device,
            )

            edge_indices, edge_weights = dense_to_sparse(cd_matrix) 

            data.n_atoms = len(atomic_numbers)
            data.pos = pos
            data.cell = cell
            data.y = torch.Tensor(np.array([target_val]))
            data.z = atomic_numbers   
            if self.name == 'cubic':
                data.AB = sdict["AB"]
            data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
            data.edge_index, data.edge_weight = edge_indices, edge_weights
            data.cell_offsets = cell_offsets

            data.edge_descriptor = {}

            data.edge_descriptor["distance"] = edge_weights
            data.distances = edge_weights
            data.structure_id = [[structure_id] * len(data.y)]


            # data.glob_feat   = sdict["gatgnn_glob_feat"]


        logging.info("Generating node features...")
        generate_node_features(data_list, self.n_neighbors, device=self.device)

        logging.info("Generating edge features...")
        generate_edge_features(data_list, self.edge_steps, self.r, device=self.device)

        # compile non-otf transforms
        logging.debug("Applying transforms.")

        # Ensure GetY exists to prevent downstream model errors
        assert self.pre_transform[0].__class__.__name__ == "GetY", "The target transform GetY is required in pre_ptransform."

        composition = Compose(self.pre_transform)

        # apply transforms
        for data in data_list:
            composition(data)

        clean_up(data_list, ["edge_descriptor"])

        return data_list