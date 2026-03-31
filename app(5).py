
import streamlit as st
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor, XGBClassifier
from gnn_model import GCNModel, combine_graphs

IC50_PATH = "ic50_model.json"
TOX_PATH = "tox_model.json"
GNN_PATH = "gnn_model.pth"

st.set_page_config(page_title="AI Drug Discovery", layout="wide")
st.title("🧪 AI Drug Discovery Platform")

@st.cache_resource
def load_models():
    ic50_model = XGBRegressor()
    ic50_model.load_model(IC50_PATH)

    tox_model = XGBClassifier()
    tox_model.load_model(TOX_PATH)

    gnn_model = GCNModel()
    gnn_model.load_state_dict(torch.load(GNN_PATH, map_location="cpu"))
    gnn_model.eval()

    return ic50_model, tox_model, gnn_model

ic50_model, tox_model, gnn_model = load_models()

morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(morgan.GetFingerprint(mol))

excipients = {
    "Lactose": "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](CO)O[C@@H]2O)[C@H](O)[C@@H](O)[C@H]1O",
    "PEG": "OCCO",
    "PVP": "C=CC(=O)N1CCCC1",
    "HPMC": "COC1=CC=CC=C1O"
}

def predict_ic50(fp):
    pic50 = ic50_model.predict(fp)[0]
    ic50 = 10 ** (9 - pic50)
    return pic50, ic50

def predict_toxicity(fp):
    pred = tox_model.predict(fp)[0]
    return "High" if pred == 1 else "Low"

def predict_compatibility(drug_smiles, exc_smiles):
    data = combine_graphs(drug_smiles, exc_smiles, 0)
    if data is None:
        return "Invalid", 0

    with torch.no_grad():
        prob = gnn_model(data).item()

    return ("Compatible" if prob > 0.5 else "Incompatible"), prob

def find_best_excipient(drug_smiles):
    best, score = None, -1
    for name, exc in excipients.items():
        _, prob = predict_compatibility(drug_smiles, exc)
        if prob > score:
            best, score = name, prob
    return best, score

smiles = st.text_input("Enter SMILES")
excipient = st.selectbox("Select Excipient", list(excipients.keys()))

if st.button("Predict"):
    fp = featurize(smiles)

    if fp is None:
        st.error("Invalid SMILES")
    else:
        fp = fp.reshape(1, -1)

        pic50, ic50 = predict_ic50(fp)
        tox = predict_toxicity(fp)
        comp, prob = predict_compatibility(smiles, excipients[excipient])
        best, score = find_best_excipient(smiles)

        col1, col2, col3 = st.columns(3)
        col1.metric("pIC50", f"{pic50:.2f}")
        col1.metric("IC50 (nM)", f"{ic50:.2f}")
        col2.metric("Toxicity", tox)
        col3.metric("Compatibility", comp)

        st.progress(prob)
        st.success(f"🏆 Best Excipient: {best} ({score:.2f})")
