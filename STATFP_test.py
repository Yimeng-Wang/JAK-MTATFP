import pandas as pd
import dgl
import torch
from torch.utils.data import DataLoader
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer

if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'


def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')

from dgllife.data import MoleculeCSVDataset
dc_listings = pd.read_csv('../MTATFP/Data/JAK_test.csv')
def load_data(data,name,load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer= bond_featurizer,
                                 smiles_column='SMILES',
                                 cache_file_path=str(name)+'_dataset.bin',
                                 task_names=['pIC50_JAK1'],
                                 load=load,init_mask=True,n_jobs=8
                            )
    return dataset
test_datasets = load_data(dc_listings,'test_ST',False)
test_loader = DataLoader(test_datasets, batch_size=256,shuffle=True,
                          collate_fn=collate_molgraphs)



#加载模型
fn = '../MTATFP/Model/STATFP_jak1.pt'
model = model_zoo.AttentiveFPPredictor(node_feat_size=n_feats,
                                   edge_feat_size=e_feats,
                                   num_layers=2,
                                   num_timesteps=1,
                                   graph_feat_size=300,
                                   n_tasks=1
                                    )
model.load_state_dict(torch.load(fn,map_location=torch.device('cpu')))
gcn_net = model.to(device)

if __name__ == '__main__':
    print('Test sets')
    test_meter=Meter()
    for batch_id, batch_data in enumerate(test_loader):
        test_smiles, test_bg, test_labels, test_masks = batch_data
        test_bg = test_bg.to(device)
        test_labels = test_labels.to(device)
        test_masks = test_masks.to(device)
        test_n_feats = test_bg.ndata.pop('hv').to(device)
        test_e_feats = test_bg.edata.pop('he').to(device)
        test_prediction = gcn_net(test_bg, test_n_feats, test_e_feats)
        test_meter.update(test_prediction, test_labels,test_masks)
    extest_r2 = test_meter.compute_metric('r2')
    extest_r2_avg = test_meter.compute_metric('r2', reduction='mean')
    extest_mae=test_meter.compute_metric('mae')
    extest_mae_avg=test_meter.compute_metric('mae', reduction='mean')
    extest_rmse=test_meter.compute_metric('rmse')
    extest_rmse_avg=test_meter.compute_metric('rmse', reduction='mean')
    print('Test_R2:',extest_r2)
    print('Test_R2:',extest_r2_avg)
    print('Test_MAE:',extest_mae)
    print('Test_MAE:',extest_mae_avg)
    print('Test_RMSE:',extest_rmse)
    print('Test_RMSE:',extest_rmse_avg)

