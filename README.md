# Lassa Virus Binding Affinity Prediction Using Concept-Bottleneck Model

A deep learning project for predicting drug-protein binding affinity (pIC50) against Lassa virus nucleoprotein using an **interpretable Concept-Bottleneck Model (CBM)** that combines multiple embedding strategies.

## 🎯 Overview

This project addresses drug discovery using **two complementary approaches** with the same CBM architecture:

### 📊 Two Datasets, One Model

**🔬 Primary: Lassa Virus Nucleoprotein (FINAL_LLM_CODE.ipynb)**
- Specialized for antiviral drug discovery
- Dataset: ~2,442 drug-protein pairs
- Optimized hyperparameters for limited data
- **Production-ready** with validated performance (R²≈0.60+)

**💊 Benchmark: DAVIS Kinase Binding (DAVIS_BINDINGAFF.ipynb)**
- General drug-target affinity prediction
- Dataset: ~13,000+ interactions (large-scale)
- Demonstrates model scalability
- Reference implementation for transfer learning

### Key Features
✅ **Concept-Bottleneck Architecture** – Extract 12 interpretable molecular/protein concepts  
✅ **Multi-modal Embeddings** – Fuse drug, protein, and chemical fingerprint representations  
✅ **Mixture of Experts** – Combine concept-based and direct prediction pathways  
✅ **Dual Dataset Support** – Works with small (Lassa) and large (DAVIS) datasets  
✅ **Production-ready** – Clean code, proper train/val/test splits, reproducible results  
✅ **PyTorch-based** – GPU-accelerated training, modular design

---

## 📓 Jupyter Notebooks (Reference Implementations)

See **`notebooks/`** folder for detailed Jupyter notebooks:

| Notebook | Dataset | Size | Focus |
|----------|---------|------|-------|
| **FINAL_LLM_CODE.ipynb** ⭐ | Lassa NP | ~2.4K pairs | Antiviral drug discovery (production-ready) |
| **DAVIS_BINDINGAFF.ipynb** | DAVIS Kinases | ~13K pairs | General drug-target affinity (benchmark) |

**Why both?**
- **FINAL_LLM_CODE** = Optimized for antiviral (your main project) ✨
- **DAVIS_BINDINGAFF** = Demonstrates model on large public dataset (validation) 📊

See `notebooks/README.md` for detailed comparison and how to run them.

**Lassa Virus Nucleoprotein (NP) Dataset:**
- **Sample size**: ~2,400–3,000 drug-protein pairs
- **pIC50 range**: 4.0 – 11.0 (log scale of IC50 in nM)
- **pIC50 statistics**: Mean ≈ 5.23, Std ≈ 1.05
- **Columns**: 
  - `pIC50` – binding affinity (target variable)
  - `Smiles` – drug chemical structure
  - `Sequence` – protein amino acid sequence
  - `Selfies` – drug structure in SELFIES notation

**Data preprocessing:**
1. Filter invalid SMILES (must parse in RDKit)
2. Remove rows with missing values
3. Filter pIC50 ∈ [4.0, 11.0]
4. Normalize pIC50 using training set mean/std
5. Split: 75% train / 10% val / 15% test

---

## 🧬 Model Architecture

### Overall Flow
```
Drug (SMILES)  ──SELFormer──┐
Protein (Seq)  ──ESM2───────┼──→ [Fuse + PCA] ──→ Fused Emb (256D)
                                                        ↓
Morgan FP      ───────────┘                         ┌────────────┐
                                                    │ CBM Model  │
                                                    └────────────┘
                                                         ↓
                                          ┌─────────────┬──────────────┐
                                          ↓             ↓              ↓
                                   Concept Extractor  CBM Head  Bypass Head
                                          ↓             ↓              ↓
                                     12 Concepts  pIC50(CBM)  pIC50(Bypass)
                                                        ↓              ↓
                                                        └──→ Mixture ←──┘
                                                              ↓
                                                         pIC50_pred
```

### Model Components

#### 1. **Embedding Encoders**
- **DrugEncoder (SELFormer)**: Converts SMILES to 768D vectors
  - Pre-trained ProtBERT model fine-tuned on molecular SMILES
  - Captures chemical properties and functional groups
  
- **ProteinEncoder (ESM2)**: Converts protein sequences to 320D vectors
  - Pre-trained ESM2 (Facebook AI) for protein understanding
  - Encodes evolutionary information and structure propensities
  
- **Morgan Fingerprints**: 1024D binary/count vectors
  - Radius-2 circular fingerprints from RDKit
  - Captures local chemical substructures

#### 2. **Fusion & Dimensionality Reduction**
- Concatenate: 768 + 320 + 1024 = **2112D**
- Apply PCA fitted on training data → **256D** (fused embeddings)
- Prevents overfitting and speeds up downstream processing

#### 3. **Concept Bottleneck Model (CBM)**

**12 Interpretable Concepts:**
| # | Concept | Type | Range |
|---|---------|------|-------|
| 0 | QED | Drug-likeness | [0, 1] |
| 1 | TPSA (normalized) | Polar surface area | [0, 1] |
| 2 | LogP | Lipophilicity | ~[-2, 6] |
| 3 | Rotatable Bonds | Flexibility | [0, 20+] |
| 4 | H-Bond Donors | Polarity | [0, 15] |
| 5 | H-Bond Acceptors | Polarity | [0, 20] |
| 6 | Molecular Weight (norm) | Size | [0, 1] |
| 7 | Aromatic Rings | Aromaticity | [0, 10] |
| 8 | Chelation Potential | Zn-binding | [0, 1] |
| 9 | Fraction CSP3 | Saturation | [0, 1] |
| 10 | Seq Length (norm) | Protein size | [0, 1] |
| 11 | Binding Compatibility | Proxy score | [0, 1] |

**Model Paths:**
1. **CBM Path**: Fused Emb → Concept Extractor → [12 concepts] → CBM Head → pIC50_CBM
2. **Bypass Path**: Fused Emb → Bypass Head → pIC50_Bypass  
3. **Mixture**: pIC50_final = α · pIC50_CBM + (1 − α) · pIC50_Bypass
   - α learned during training (mixture weight)

#### 4. **Loss Function**
```
Loss = MSE(pIC50_pred, pIC50_true) + λ · |Concepts| + Regularization
```
- **λ (concept_weight)**: 0.3 (encourage sparse, interpretable concepts)
- **Gradient clipping**: max_norm=1.0 (stable training)

#### 5. **Training Details**
- **Optimizer**: AdamW with 3 learning rates:
  - Concept extractor & CBM head: 1e-4
  - Bypass head: 3e-4 (3× higher to converge faster)
- **Scheduler**: CosineAnnealingWarmRestarts (T₀=20, T_mult=2)
- **Dropout**: 0.25 (reduced from 0.35 to combat underfitting)
- **Batch size**: 32
- **Epochs**: 150 (early stopping via validation R²)

---

## 📈 Results

### Test Set Performance
| Metric | Value |
|--------|-------|
| **RMSE** | ~0.85 pIC50 units |
| **R²** | ~0.58–0.65 |
| **Pearson r** | ~0.77–0.80 |
| **Spearman ρ** | ~0.75–0.78 |
| **Num Test Samples** | ~400 |

**Interpretation:**
- Model explains **60%+ of variance** in binding affinity
- Predictions correlate strongly with ground truth (~0.77–0.80)
- RMSE of ~0.85 log10(nM) ≈ ±7-fold error in IC50 concentrations

---

## 🚀 Installation & Setup

### Prerequisites
- **Python** 3.8+
- **CUDA** 11.8+ (optional, for GPU acceleration; CPU also works)
- **Git**

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/lassa-binding-affinity.git
cd lassa-binding-affinity
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n lassa python=3.10
conda activate lassa
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Installation time:** ~5–10 minutes (depending on PyTorch build)

### Step 4: Prepare Data
Place your data file at:
```
./data/lassa_clean.csv
```

Expected columns:
```
pIC50, Smiles, sequence, selfies
```

### Step 5: Precompute Embeddings (Optional)
If you have pre-computed embeddings, place them in:
```
./embeddings/
├── drug_embs.npy      (N × 768)
├── prot_embs.npy      (N × 320)
├── morgan_embs.npy    (N × 1024)
└── pca_model.pkl      (fitted PCA)
```

---

## 💻 Usage

### Option A: Training from Scratch
```bash
python main.py
```

**Output:**
- `./models/best_model.pt` – Best model checkpoint
- `./outputs/results.json` – Test metrics (RMSE, R², correlations)
- `./embeddings/pca_model.pkl` – Trained PCA model

### Option B: Custom Configuration
Edit `main.py` or create a config file:
```python
from main import Config

cfg = Config(
    csv_path='data/lassa_clean.csv',
    batch_size=64,
    num_epochs=200,
    learning_rate=5e-5,
    dropout=0.2
)
```

### Option C: Load Trained Model & Make Predictions
```python
import torch
from main import ImprovedCBMModel, Config
import numpy as np

cfg = Config()
model = ImprovedCBMModel(cfg)
model.load_state_dict(torch.load('./models/best_model.pt'))
model.eval()

# Dummy fused embedding
fused_emb = torch.randn(1, cfg.fused_dim)

with torch.no_grad():
    pIC50_pred, concepts = model(fused_emb)
    print(f"Predicted pIC50: {pIC50_pred.item():.2f}")
    print(f"Concepts: {concepts.squeeze().numpy()}")
```

---

## 📁 Project Structure
```
lassa-binding-affinity/
├── README.md                           # This file (main documentation)
├── main.py                             # Production Python code
├── requirements.txt                    # Dependencies
├── GITHUB_SETUP.md                     # GitHub push guide
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore patterns
│
├── notebooks/                          # ⭐ Jupyter reference implementations
│   ├── README.md                       # Notebook guide & comparison
│   ├── FINAL_LLM_CODE.ipynb            # Lassa virus (main project)
│   └── DAVIS_BINDINGAFF.ipynb          # DAVIS kinases (validation)
│
├── data/                               # Input data (add this!)
│   └── lassa_clean.csv                 # Your dataset (not uploaded)
│
├── models/                             # Trained model checkpoints (generated)
│   └── best_model.pt
│
├── embeddings/                         # Pre-computed embeddings (optional)
│   ├── drug_embs.npy
│   ├── prot_embs.npy
│   ├── morgan_embs.npy
│   └── pca_model.pkl
│
└── outputs/                            # Results & metrics (generated)
    └── results.json
```

---

## 🔍 Concept Interpretation

The extracted concepts provide **human-interpretable explanations**:

**Example Output:**
```
Predicted pIC50: 6.45

Concepts:
  0. QED (drug-likeness):         0.82 ✓ Good
  1. TPSA (polarity):              0.35 ✓ Moderate
  2. LogP (lipophilicity):          2.1  ✓ Optimal
  3. Rotatable bonds (flexibility): 5.0  ✓ Good
  4. H-donors:                      2.0  ✓ Normal
  5. H-acceptors:                   4.0  ✓ Normal
  6. MW (normalized):               0.65 ✓ Good
  7. Aromatic rings:                2.0  ✓ Good
  8. Chelation potential:           0.15 → Low Zn-binding
  9. Frac CSP3:                      0.35 → Relatively saturated
  10. Protein length (norm):        0.8  ✓ Normal
  11. Binding compatibility:        0.65 ✓ Good

Prediction confidence: High (bypass weight ≈ 0.4, CBM weight ≈ 0.6)
```

**Use cases:**
- **Drug optimization**: Adjust LogP, TPSA, or rotatable bonds to improve affinity
- **SAR analysis**: Understand which molecular features drive binding
- **Lead prioritization**: Filter compounds by concept scores

---

## 🛠️ Troubleshooting

### Issue: CUDA out of memory
```bash
# Reduce batch size in main.py
cfg.batch_size = 16
```

### Issue: RDKit import error
```bash
conda install -c conda-forge rdkit
```

### Issue: SELFormer/ESM2 download fails
```bash
# Download models manually
from transformers import AutoModel
AutoModel.from_pretrained('nferruz/ProtBERT_SMILES')
AutoModel.from_pretrained('facebook/esm2_t6_8M_UR50D')
```

### Issue: Data file not found
```
Ensure ./data/lassa_clean.csv exists with columns:
pIC50, Smiles, sequence, selfies
```

---

## 📚 References

**Embedding Models:**
- SELFormer: [ProtBERT](https://huggingface.co/nferruz/ProtBERT_SMILES)
- ESM2: [Facebook AI](https://www.science.org/doi/10.1126/science.ade2574)
- Morgan Fingerprints: [RDKit](https://www.rdkit.org/)

**Methods:**
- Concept Bottleneck Models: [Koh et al., ICML 2020](https://arxiv.org/abs/2007.04871)
- Binding Affinity Prediction: [Öztürk et al., Bioinformatics 2018](https://academic.oup.com/bioinformatics/article/34/17/i829/5093245)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** – see LICENSE file for details.

---

## ✉️ Contact & Citation

**Author**: [Your Name]  
**Email**: [your.email@example.com]  
**Lab/Affiliation**: [Your Lab/University]

If you use this code in research, please cite:
```bibtex
@software{lassa_cbm_2024,
  author = {Your Name},
  title = {Lassa Virus Binding Affinity Prediction with Interpretable CBM},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/lassa-binding-affinity}
}
```

---

## 🎓 Key Learnings

This project demonstrates:
- ✅ **Multi-modal fusion**: Combining heterogeneous embeddings (SMILES, sequences, fingerprints)
- ✅ **Interpretability**: Using concept bottleneck models for explainable predictions
- ✅ **PyTorch best practices**: Modular design, proper train/val/test splits, scheduler usage
- ✅ **Drug discovery ML**: Real-world molecular property prediction
- ✅ **Reproducibility**: Fixed seeds, documented hyperparameters, clean code

---

**Last Updated**: July 2024  
**Status**: ✅ Ready for production use
