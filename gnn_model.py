
import torch
import torch.nn as nn
from torch_geometric.data import Data
from rdkit import Chem

# ===============================
# SIMPLE GCN MODEL
# ===============================
class GCNModel(nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, data):
        # Dummy forward (since trained model is loaded)
        return torch.sigmoid(torch.tensor([0.5]))

# ===============================
# GRAPH CREATION FUNCTION
# ===============================
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()

    x = torch.ones((num_atoms, 1))

    edge_index = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index).t().contiguous()

    return Data(x=x, edge_index=edge_index)

# ===============================
# COMBINE DRUG + EXCIPIENT
# ===============================
def combine_graphs(smiles1, smiles2, label):
    g1 = smiles_to_graph(smiles1)
    g2 = smiles_to_graph(smiles2)

    if g1 is None or g2 is None:
        return None

    # Simple merge (dummy for compatibility)
    x = torch.cat([g1.x, g2.x], dim=0)
    edge_index = torch.cat([g1.edge_index, g2.edge_index], dim=1)

    return Data(x=x, edge_index=edge_index, y=torch.tensor([label]))
