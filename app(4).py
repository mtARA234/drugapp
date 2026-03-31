import streamlit as st
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator
from xgboost import XGBRegressor, XGBClassifier
import shap
import matplotlib.pyplot as plt

# ===============================
# IMPORT GNN
# ===============================
from gnn_model import GCNModel, combine_graphs

# ===============================
# PATHS (adjust these to your Google Drive paths)
# ===============================
IC50_PATH = "/content/drive/MyDrive/drug_app/models/ic50_model.json"
TOX_PATH = "/content/drive/MyDrive/drug_app/models/tox_model.json"
GNN_PATH = "/content/drive/MyDrive/drug_app/models/gnn_model.pth"

# ===============================
# UI
# ===============================
st.set_page_config(layout="wide")
st.title("🧪 AI Drug Discovery Platform")

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    ic50 = XGBRegressor()
    ic50.load_model(IC50_PATH)

    tox = XGBClassifier()
    tox.load_model(TOX_PATH)

    gnn = GCNModel()
    gnn.load_state_dict(torch.load(GNN_PATH, map_location="cpu"))
    gnn.eval()

    return ic50, tox, gnn

ic50_model, tox_model, gnn_model = load_models()

# ===============================
# FINGERPRINT
# ===============================
morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, np.zeros(1024)
    return mol, np.array(morgan.GetFingerprint(mol))

def featurize_with_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    bit_info = {}
    fp = morgan.GetFingerprint(mol, bitInfo=bit_info)
    return mol, np.array(fp), bit_info

# ===============================
# EXCIPIENTS
# ===============================
excipients = {
    "Lactose": "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](CO)O[C@@H]2O)[C@H](O)[C@@H](O)[C@H]1O",
    "PEG": "OCCO",
    "PVP": "C=CC(=O)N1CCCC1",
    "HPMC": "COC1=CC=CC=C1O"
}

# ===============================
# PREDICTION FUNCTIONS
# ===============================
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

# ===============================
# SHAP / Atom Highlight
# ===============================
explainer = shap.TreeExplainer(tox_model)

def highlight_atoms(smiles):
    mol, fp, bit_info = featurize_with_info(smiles)
    if mol is None:
        return None

    fp = fp.reshape(1, -1)
    shap_values = explainer.shap_values(fp)
    important_bits = np.argsort(np.abs(shap_values[0]))[-10:]

    atoms = set()
    for bit in important_bits:
        if bit in bit_info:
            for atom_id, radius in bit_info[bit]:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_id)
                for b in env:
                    bond = mol.GetBondWithIdx(b)
                    atoms.add(bond.GetBeginAtomIdx())
                    atoms.add(bond.GetEndAtomIdx())

    # Use Matplotlib-based drawing (works in Colab)
    fig = Draw.MolToMPL(mol, highlightAtoms=list(atoms))
    return fig

# ===============================
# BEST EXCIPIENT FUNCTION
# ===============================
def find_best_excipient(drug_smiles):
    best_excipient = None
    best_score = -1
    for name, exc_smiles in excipients.items():
        label, prob = predict_compatibility(drug_smiles, exc_smiles)
        if prob > best_score:
            best_score = prob
            best_excipient = name
    return best_excipient, best_score

# ===============================
# UI INPUT
# ===============================
col1, col2 = st.columns(2)

with col1:
    smiles = st.text_input("Enter Drug SMILES")

with col2:
    excipient = st.selectbox("Select Excipient", list(excipients.keys()))

# ===============================
# RUN PREDICTION
# ===============================
if st.button("Run Prediction"):
    with st.spinner("Running AI models..."):
        mol, fp = featurize(smiles)

        if mol is None:
            st.error("Invalid SMILES")
        else:
            fp = fp.reshape(1, -1)
            pic50, ic50 = predict_ic50(fp)
            tox = predict_toxicity(fp)
            comp, prob = predict_compatibility(smiles, excipients[excipient])
            best_exc, best_score = find_best_excipient(smiles)

            # ===============================
            # DISPLAY RESULTS
            # ===============================
            st.subheader("📊 Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("pIC50", f"{pic50:.2f}")
            c1.metric("IC50 (nM)", f"{ic50:.2f}")
            c2.metric("Toxicity", tox)
            c3.metric("Compatibility", comp)
            st.progress(prob)
            st.success(f"🏆 Best Excipient: {best_exc} (Score: {best_score:.2f})")

            # Show atom highlighting
            fig = highlight_atoms(smiles)
            if fig:
                st.pyplot(fig)
