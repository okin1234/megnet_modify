import os
import tensorflow as tf
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from itertools import islice
import torch

import warnings
''' 다시 경고메세지들 보고 싶으면 ignore -> default '''
warnings.filterwarnings("ignore")

from megnet.data.molecule import MolecularGraph
from megnet.data.molecule import SimpleMolGraph
from megnet.data.crystal import CrystalGraph

from megnet.callbacks import ReduceLRUponNan, ManualStop
from megnet.data.graph import GraphBatchDistanceConvert, GaussianDistance
from megnet.models import MEGNetModel

import openbabel
from openbabel.pybel import readstring
ob_log_handler = openbabel.pybel.ob.OBMessageHandler()
#ob_log_handler.SetOutputLevel(0) # only critical message printing (warning message is passing)
openbabel.pybel.ob.obErrorLog.StopLogging() # all error is not printing



class make_datasets():
    def __init__(self, base_dir):
        self.base_dir = base_dir
    
    def data_to_onehot(self, rdkit_data):
        from sklearn.preprocessing import LabelBinarizer

        for n, c in enumerate(rdkit_data.columns[0:-1]):
            try:
                if n==0:
                    total_onehot_data = LabelBinarizer().fit_transform(rdkit_data[c])
                onehot_data = LabelBinarizer().fit_transform(rdkit_data[c])
                total_onehot_data = np.concatenate([total_onehot_data, onehot_data], axis=1)
            except:
                continue
        rdkit_data_onehot = {}
        for n, cid in enumerate(rdkit_data['cid'].to_list()):
            rdkit_data_onehot[cid] = total_onehot_data[n]
        return rdkit_data_onehot

    def data_sampling(self, input_df=None, output_df=None, mode='qcut', sample_column=None, class_num=5):
        if mode=='cut':
            output_df[sample_column] = pd.cut(input_df[sample_column], class_num, duplicates='drop', labels=False)
        elif mode=='qcut':
            output_df[sample_column] = pd.qcut(input_df[sample_column], class_num, duplicates='drop', labels=False)

        ''' class별 elements 개수 '''
        class_counts = output_df[sample_column].value_counts()

        return output_df, class_counts

    def feature_to_class(self, xtb_df, g09_df, to_class=False, class_num=5, cut_mode='qcut'):
        xtb_features = ['cid']
        g09_features = ['cid']
        xtb_features.extend(xtb_df.columns[45:])
        g09_features.extend(g09_df.columns[30:])

        xtb_df = xtb_df[xtb_features]
        g09_df = g09_df[g09_features]
        g09_cid_list = g09_df['cid'].to_list()
        xtb_df = xtb_df[~xtb_df['cid'].isin(g09_cid_list)]  
        feature_df = pd.concat([xtb_df, g09_df])

        if to_class:
            final_df=pd.DataFrame([])
            final_df['cid'] = feature_df['cid']
            for feature in xtb_features[1:]:
                final_df, class_counts = self.data_sampling(input_df=feature_df, output_df=final_df, mode=cut_mode, sample_column=feature, class_num=class_num)
        else:
            final_df = feature_df
        return final_df

    def extract_rdkit_descriptor(self, importance_rank, to_class, class_num, cut_mode):
        lightgbm_importance = pd.read_csv(os.path.join(self.base_dir, 'Boosting_model/feature_importance_with_recalc_data.csv'))
        selected_feature_list = lightgbm_importance['features'][:importance_rank].to_list()
        selected_feature_list.extend(['cid'])

        g09_df = pd.read_csv(os.path.join(self.base_dir, 'Boosting_model/training_2d_descriptor_data_rdkit.csv'))
        xtb_df = pd.read_csv(os.path.join(self.base_dir, 'Boosting_model/xtb_rdkit_2d_descriptor_data.csv'))
        rdkit_data = self.feature_to_class(xtb_df, g09_df, to_class=to_class, class_num=class_num, cut_mode=cut_mode)

        rdkit_data = rdkit_data[selected_feature_list]
        return rdkit_data

    def add_rdkit_descriptor(self, name, state_attributes, rdkit_data, onehot=False):
        cid = int(name)
        if onehot:
            rdkit_data_cid = rdkit_data[cid]
            state_attributes[0].extend(rdkit_data_cid)
        else:
            rdkit_data_cid = rdkit_data[rdkit_data['cid']==cid]
            rdkit_data_cid = rdkit_data_cid.iloc[0].to_list()[:-1]
            state_attributes[0].extend(rdkit_data_cid)

        return state_attributes

    def make_graph_datas(self, num_xtb_data, num_g09_data, atom_features, bond_features, load=False, random_state=1000, g09_data_cut=None, with_rdkit=False, rdkit_rank=10, to_class=False, class_num=5, cut_mode='qcut', onehot=False):
        def xyz2graph(xyz_file, state_attributes):
            with open(xyz_file, 'r') as f:
                string = f.read()
            name = os.path.basename(xyz_file).split(".xyz")[0]
            mol = readstring('xyz', string)
            structure_data = molecule_graph.convert(mol, state_attributes=state_attributes, full_pair_matrix=False)
            return name, structure_data

        ''' graph convert model defination '''
        # relaxed xyz file list
        xtb_xyz_list = glob(os.path.join(self.base_dir, "xtb/xyz_files/*.xyz"))
        g09_xyz_list = glob(os.path.join(self.base_dir, "g09/relaxed_xyz/*.xyz"))

        # graph converter defination
        known_elements = ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'S', 'P', 'I', 'Si', 'Li', 'B', 'Ge']
        molecule_graph = MolecularGraph(atom_features=atom_features, bond_features=bond_features, known_elements=known_elements)

        if not load:
            ''' structure graph generation '''
            g09_datas = {}
            print('g09 datas convert to graph datas~~')
            ''' rdkit important feature load '''
            rdkit_data = self.extract_rdkit_descriptor(importance_rank=rdkit_rank, to_class=to_class, class_num=class_num, cut_mode=cut_mode)
            if onehot:
                rdkit_data = self.data_to_onehot(rdkit_data)

            for xyz in tqdm(g09_xyz_list):
                state_attributes = [[1]] # xtb=0, g09=1
                name = os.path.basename(xyz).split(".xyz")[0]
                data_type = 'g09'
                if with_rdkit:
                    state_attributes = self.add_rdkit_descriptor(name, state_attributes, rdkit_data, onehot=onehot)

                name, structure_data = xyz2graph(xyz, state_attributes)
                name = name+"_g09"
                g09_datas[name] = structure_data

            ''' graph data save '''
            import pickle
            with open('g09_datas.pickle','wb') as fw:
                pickle.dump(g09_datas, fw)

            #with open('xtb_datas.pickle', 'wb') as fw:
            #    pickle.dump(xtb_datas, fw)
        else:
            ''' structure graph load '''
            import pickle
            with open('g09_datas.pickle','rb') as fr:
                g09_datas = pickle.load(fr)

            #with open('xtb_datas.pickle', 'rb') as fr:
            #    xtb_datas = pickle.load(fr)

        ''' target data generation '''
        xtb_result_csv = pd.read_csv(os.path.join(self.base_dir, "xtb/xtb_crest_result_addit.csv"))
        g09_result_csv = pd.read_csv(os.path.join(self.base_dir, "g09/g09_selected_mols_result.csv"))
        if not g09_data_cut == None:
            g09_result_csv = g09_result_csv.iloc[:g09_data_cut]

        if not num_g09_data == None:
            g09_result_csv = g09_result_csv.sample(n=num_g09_data, random_state=random_state)

        ''' xtb error data 제거 '''
        xtb_error_cid_1 = pd.read_csv(os.path.join(self.base_dir, "xtb/error_cid_list.csv"))
        xtb_error_cid_1 = xtb_error_cid_1['error_cid'].to_list()
        with open(os.path.join(self.base_dir, "xtb/charge_error_cid_list.pickle"), 'rb') as fr:
            xtb_error_cid_2 = pickle.load(fr)
        xtb_error_cid_1.extend(xtb_error_cid_2)
        xtb_result_csv = xtb_result_csv[~xtb_result_csv['cid'].isin(xtb_error_cid_1)]

        ''' overftting 방지를 위해 xtb data에서 g09 data 제거 '''
        xtb_result_csv = xtb_result_csv[~xtb_result_csv['cid'].isin(g09_result_csv['cid'].tolist())]

        g09_target = {}
        print('g09 datas convert to graph datas~~')
        for i in tqdm(range(len(g09_result_csv))):
            name = str(int(g09_result_csv.iloc[i]['cid']))+"_g09"
            voltage = g09_result_csv.iloc[i]['voltage']
            g09_target[name] = voltage

        xtb_target = {}
        print('xtb datas conver to graph datas~~ (it may be doing for few minutes...)')
        for i in tqdm(range(len(xtb_result_csv))):
            name = str(int(xtb_result_csv.iloc[i]['cid']))+"_xtb"
            voltage = xtb_result_csv.iloc[i]['voltage_avg_real']
            xtb_target[name] = voltage


        ''' target and graph matching (graph data에는 target에 없는 구조도 있음 (error 구조가 제거 안되어 있어서 그럼) 이걸 보정하기 위한 작업 '''
        ''' g09_datas cleaning '''
        g09_key_list = g09_target.keys()
        for key, item in list(g09_datas.items()):
            if key not in g09_key_list:
                g09_datas.pop(key, None)
        print('The length of g09_datas is', len(g09_datas))
        print('The length of g09_targets is', len(g09_target))

        ''' xtb data 갯수 조절 '''
        import random
        random.seed(random_state)
        if num_xtb_data is None:
            num_xtb_data = len(xtb_target)
        xtb_keys = random.sample(xtb_target.keys(), num_xtb_data)
        #xtb_datas = {k: xtb_datas[k] for k in xtb_keys}
        xtb_target = {k: xtb_target[k] for k in xtb_keys}

        ''' xtb graph data 생성 '''
        if not load:
            xtb_datas = {}
            print('xtb datas conver to graph datas~~ (it may be doing for few minutes...)')
            for xyz in tqdm(xtb_xyz_list):
                key_name = os.path.basename(xyz).split(".xyz")[0]+"_xtb"
                if key_name in xtb_keys:
                    state_attributes = [[0]] # xtb=0, g09=1
                    name = os.path.basename(xyz).split(".xyz")[0]
                    if with_rdkit:
                        try:
                            state_attributes = self.add_rdkit_descriptor(name, state_attributes, rdkit_data, onehot=onehot)
                        except:
                            xtb_target.pop(key_name)
                            continue

                    name, structure_data = xyz2graph(xyz, state_attributes)
                    name = name+"_xtb"
                    xtb_datas[name] = structure_data
            ''' graph data save '''
            import pickle
            with open('xtb_datas.pickle','wb') as fw:
                pickle.dump(xtb_datas, fw)
        else:
            import pickle
            with open('xtb_datas.pickle','rb') as fr:
                xtb_datas = pickle.load(fr)
        print('The length of xtb_datas is', len(xtb_datas))
        print('The length of xtb_targets is', len(xtb_target))  

        ''' g09 data and xtb data merging for training '''
        final_graphs = dict(xtb_datas, **g09_datas)
        final_targets = dict(xtb_target, **g09_target)
        material_ids = list(g09_datas.keys())+list(xtb_datas.keys())


        ''' dataset splits by state feature random distribute (by calculation fidelity) '''
        from sklearn.model_selection import train_test_split

        ##  train:val = 8:2
        fidelity_list = [i.split('_')[1] for i in material_ids]
        train_ids, val_ids = train_test_split(material_ids, stratify=fidelity_list, 
                                                   test_size=0.2, random_state=random_state)

        ''' remove xtb from validation (for prevent overfitting to xtb data) '''
        val_ids_g09 = [i for i in val_ids if not i.endswith('xtb')]
        val_ids_xtb = [i for i in val_ids if not i.endswith('g09')]
        val_ids = val_ids_g09
        ''' validation에서 xtb 안쓰는데 8:2로만 나눠도 2만개 넘게 데이터 날라감 아까우니까 다시 training에 추가 '''
        train_ids = train_ids + val_ids_xtb

        print("Train, val data sizes are ", len(train_ids), len(val_ids))
        train_num = len(train_ids)
        val_num = len(val_ids)


        ''' Get the train, val and test graph-target pairs '''
        def get_graphs_targets(ids):
            """
            Get graphs and targets list from the ids

            Args:
                ids (List): list of ids

            Returns:
                list of graphs and list of target values
            """
            ids = [i for i in ids if i in final_graphs]
            return [final_graphs[i] for i in ids], [final_targets[i] for i in ids]

        train_graphs, train_targets = get_graphs_targets(train_ids)
        val_graphs, val_targets = get_graphs_targets(val_ids)

        return train_graphs, train_targets, val_graphs, val_targets