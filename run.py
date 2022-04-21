import os
import tensorflow as tf
import torch
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from itertools import islice
from megnet.data.molecule import MolecularGraph
from megnet.data.molecule import MolecularGraph
from megnet.data.molecule import SimpleMolGraph
from megnet.data.crystal import CrystalGraph
from megnet.callbacks import ReduceLRUponNan, ManualStop
from megnet.data.graph import GraphBatchDistanceConvert, GaussianDistance
from megnet.models import MEGNetModel
import dataset
from dataset import make_datasets

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
class trainer():
    def __init__(self, base_dir, 
                 atom_features=['element', 'formal_charge', 'hybridization', 'donor', 'acceptor'], bond_features=['spatial_distance', 'same_ring', 'bond_type'], global_features=['fedility'],
                 nblocks=3, global_embedding_dim=16, npass=2,
                num_xtb_data=None, num_g09_data=None, load=False, random_state=1000, g09_data_cut=None, with_rdkit=False, rdkit_rank=10, to_class=False, class_num=5, cut_mode='qcut', onehot=False):
        self.base_dir = base_dir
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.global_features = global_features
        self.nblocks = nblocks
        self.global_embedding_dim = global_embedding_dim
        self.npass = npass
        
        self.num_xtb_data = num_xtb_data
        self.num_g09_data = num_g09_data
        self.load = load
        self.random_state = random_state
        self.g09_data_cut = g09_data_cut
        self.with_rdkit = with_rdkit
        self.rdkit_rank = rdkit_rank
        self.to_class = to_class
        self.class_num = class_num
        self.cut_mode = cut_mode
        self.onehot = onehot
        
        
    def make_train_val_set(self):
        known_elements = ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'S', 'P', 'I', 'Si', 'Li', 'B', 'Ge']
        molecule_graph = MolecularGraph(atom_features=self.atom_features, bond_features=self.bond_features, known_elements=known_elements)
        dataset_maker = make_datasets(self.base_dir)
        ''' make graph datas '''
        train_graphs, train_targets, val_graphs, val_targets = dataset_maker.make_graph_datas(self.num_xtb_data, self.num_g09_data, self.atom_features, self.bond_features, 
                                                                             load=self.load, random_state=self.random_state, g09_data_cut=self.g09_data_cut,
                                                                               with_rdkit=self.with_rdkit, rdkit_rank=self.rdkit_rank, 
                                                                                to_class=self.to_class, class_num=self.class_num, cut_mode=self.cut_mode, onehot=self.onehot)
        return train_graphs, train_targets, val_graphs, val_targets


    def train(self, epochs=200, lr=1e-3, batch_size=128, dropout=None, 
             wandb_on=True, wandb_project=None, wandb_name=None, wandb_memo=None, wandb_group=None):

        ''' 기타 변수 정의 '''
        #tf.compat.v1.disable_eager_execution()
        GPU_INDEX = '0'
        tf.random.set_seed(self.random_state)
        #os.environ['CUDA_VISIBLE_DEVICES']  = '-1'
        
        known_elements = ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'S', 'P', 'I', 'Si', 'Li', 'B', 'Ge']
        molecule_graph = MolecularGraph(atom_features=self.atom_features, bond_features=self.bond_features, known_elements=known_elements)
        train_graphs, train_targets, val_graphs, val_targets = self.make_train_val_set()
        

        train_num = len(train_targets)
        val_num = len(val_targets)

        ''' wandb settings '''
        if wandb_on:
            import wandb
            from wandb.keras import WandbCallback

            wandb.init(project=wandb_project, reinit=True, group=wandb_group, notes=wandb_memo)
            if wandb_project is None:
                wandb.init(project="testing", reinit=True, group=wandb_group, notes=wandb_memo)
            wandb.run.name = wandb_name
            wandb.run.save()

            parameters = wandb.config
            parameters.epochs = epochs
            parameters.nblocks = self.nblocks
            parameters.lr = lr
            parameters.batch_size = batch_size
            parameters.dropout = dropout
            parameters.atom_features = self.atom_features
            parameters.bond_features = self.bond_features
            parameters.global_features = self.global_features
            parameters.global_embedding_dim = self.global_embedding_dim
            parameters.npass = self.npass
            parameters.train_num = train_num
            parameters.val_num = val_num
            parameters.with_rdkit = self.with_rdkit
            if self.with_rdkit:
                parameters.rdkit_rank = self.rdkit_rank
                if self.to_class:
                    parameters.to_class = self.to_class
                    parameters.class_num = self.class_num
                    parameters.cut_mode = self.cut_mode

        format_ = val_graphs[0]
        num_atom_fea = torch.tensor(format_['atom']).shape[1]
        num_bond_fea = torch.tensor(format_['bond']).shape[1]
        num_state_fea = torch.tensor(format_['state']).shape[1]

        if wandb_on:
            parameters.num_atom_fea = num_atom_fea
            parameters.num_bond_fea = num_bond_fea
            parameters.num_state_fea = num_state_fea

        nfeat_node = num_atom_fea
        nfeat_edge = num_bond_fea
        nfeat_global = num_state_fea
        ''' model defination '''

        if wandb_on:
            callbacks = [ReduceLRUponNan(patience=200), ManualStop(), WandbCallback(monitor="val_mae", log_evaluation=True)]
        else:
            callbacks = [ReduceLRUponNan(patience=200), ManualStop()]

        model = MEGNetModel(nfeat_edge=nfeat_edge, nfeat_global=nfeat_global, nfeat_node=nfeat_node,
                            global_embedding_dim=self.global_embedding_dim,  nblocks=self.nblocks, dropout=dropout, 
                            npass=self.npass, graph_converter=molecule_graph, learning_rate=lr, batch_size=batch_size, metrics=['mae'])

        ''' training '''
        EPOCHS = epochs
        model.train_from_graphs(train_graphs, train_targets, val_graphs, val_targets,  
                                epochs=EPOCHS, verbose=2, initial_epoch=0, callbacks=callbacks)
        wandb.finish()

    def eval(selected_model=None, xtb_state=0, g09_state=1, batch_size=128, dropout=None):

        ''' 기타 변수 정의 '''
        #tf.compat.v1.disable_eager_execution()
        GPU_INDEX = '0'
        tf.random.set_seed(self.random_state)
        #os.environ['CUDA_VISIBLE_DEVICES']  = '-1'
        train_graphs, train_targets, val_graphs, val_targets = self.make_train_val_set()

        train_data = model.graph_converter.get_flat_data(train_graphs, train_targets)
        val_data = model.graph_converter.get_flat_data(val_graphs, val_targets)

        train_gen = GraphBatchDistanceConvert(*train_data, distance_converter=model.graph_converter.bond_converter, batch_size=batch_size)
        val_gen = GraphBatchDistanceConvert(*val_data, distance_converter=model.graph_converter.bond_converter, batch_size=batch_size)

        train_num = len(train_targets)
        val_num = len(val_targets)

        format_ = val_graphs[0]
        num_atom_fea = torch.tensor(format_['atom']).shape[1]
        num_bond_fea = torch.tensor(format_['bond']).shape[1]
        num_state_fea = torch.tensor(format_['state']).shape[1]

        nfeat_node = num_atom_fea
        nfeat_edge = num_bond_fea
        nfeat_global = num_state_fea
        ''' model defination '''

        model = MEGNetModel(nfeat_edge=nfeat_edge, nfeat_global=nfeat_global, nfeat_node=nfeat_node,
                            global_embedding_dim=self.global_embedding_dim,  nblocks=self.nblocks, dropout=dropout, 
                            npass=self.npass, graph_converter=molecule_graph, learning_rate=lr, batch_size=batch_size, metrics=['mae'])

        model.load_weights(selected_model)

        ''' evaluate '''

        train_preds = []
        train_trues = []
        val_preds = []
        val_trues = []

        for i in range(len(train_gen)):
            d = train_gen[i]
            train_preds.extend(model.predict(d[0]).ravel().tolist())
            train_trues.extend(d[1].ravel().tolist())

        for i in range(len(val_gen)):
            d = val_gen[i]
            val_preds.extend(model.predict(d[0]).ravel().tolist())
            val_trues.extend(d[1].ravel().tolist())

        return train_preds, train_trues, val_preds, val_trues

        