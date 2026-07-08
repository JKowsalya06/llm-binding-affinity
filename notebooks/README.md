# Jupyter Notebooks - Reference Implementations

This folder contains the original Jupyter notebooks used to develop and test the Concept-Bottleneck Model (CBM) for drug-binding affinity prediction.

## 📓 Notebooks

### 1. **DAVIS_BINDINGAFF.ipynb** 
General drug-target binding affinity prediction using the **DAVIS dataset**

**Dataset:**
- Source: DAVIS kinase inhibitor dataset
- Size: ~13,000+ drug-protein pairs (large-scale)
- Target: Kinase binding affinity
- Format: Standard SMILES + protein sequences

**Model Architecture:**
- SELFormer (drug encoding)
- ESM2 (protein encoding)
- Morgan fingerprints (chemical features)
- Concept bottleneck layer (interpretability)

**Key Features:**
- General drug discovery approach
- Suitable for kinase inhibitors
- Scalable to large datasets
- Cross-domain applicability

**Use Case:** When you have diverse drug-protein interactions or working with kinase targets

---

### 2. **FINAL_LLM_CODE.ipynb** ⭐ (Main Implementation)
Specialized Lassa virus nucleoprotein binding affinity prediction

**Dataset:**
- Source: Lassa virus nucleoprotein (NP)
- Size: ~2,442 drug-protein pairs
- Target: Antiviral compound screening
- Format: SMILES + sequences + SELFIES

**Model Architecture:**
- Improved CBM with dropout optimization (0.25)
- Residual connections in bypass path
- Balanced mixture-of-experts weighting
- Concept regularization (λ=0.3)

**Key Features:**
- Optimized for Lassa virus drug discovery
- Better handling of limited data
- Improved concept extraction
- Validated metrics (R²≈0.60+, RMSE≈0.85)

**Use Case:** Antiviral drug development, pandemic preparedness

---

## 🔄 Comparison

| Aspect | DAVIS | Lassa |
|--------|-------|-------|
| **Dataset Size** | Large (~13K) | Medium (~2.4K) |
| **Domain** | General kinases | Antiviral/Lassa |
| **Data Scarcity** | Less of an issue | Needs regularization |
| **Model Focus** | Scalability | Interpretability |
| **Dropout** | Standard (0.35) | Optimized (0.25) |
| **Use Case** | Drug discovery (broad) | Lassa drug screening |
| **Complexity** | High | Balanced |

---

## 🚀 How to Use These Notebooks

### Run in Google Colab (Recommended)
1. Upload `.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Connect to Google Drive for data storage
3. Run cells sequentially (⏯️ Shift+Enter)
4. Modify paths & hyperparameters as needed

### Run Locally
1. Ensure Jupyter is installed: `pip install jupyter`
2. Open notebook: `jupyter notebook FINAL_LLM_CODE.ipynb`
3. Install dependencies: `!pip install -r requirements.txt`
4. Prepare data at `./data/lassa_clean.csv`
5. Run cells

### Extract Code to Python Script
Convert notebook to Python:
```bash
jupyter nbconvert --to script FINAL_LLM_CODE.ipynb
```

---

## 📝 Development Workflow

These notebooks represent the **iterative development process**:

1. **Exploration** (DAVIS_BINDINGAFF.ipynb)
   - Test on general dataset
   - Establish baseline
   - Validate architecture

2. **Optimization** (FINAL_LLM_CODE.ipynb)
   - Adapt to Lassa dataset
   - Improve for limited data
   - Enhance interpretability
   - Achieve production-ready performance

---

## 🔧 Modifications for Your Project

Want to adapt these notebooks?

### For New Dataset
```python
# Replace CSV path
csv_path = 'data/your_dataset.csv'

# Update expected columns
assert all(col in df.columns for col in ['binding_affinity', 'smiles', 'sequence'])

# Adjust pIC50 range if different
df = df[df['binding_affinity'].between(min_val, max_val)]
```

### Adjust Hyperparameters
```python
cfg = Config(
    batch_size=32,        # Increase if GPU memory available
    num_epochs=200,       # More epochs for small datasets
    learning_rate=1e-4,   # Lower LR for stable training
    dropout=0.25,         # Increase if overfitting
    concept_weight=0.3    # Higher = more concept emphasis
)
```

### Change Embedding Models
```python
cfg.selformer_id = 'other/smiles-bert'    # Alternative drug encoder
cfg.esm_id = 'facebook/esm2_t12_35M_UR50D'  # Different ESM2 size
```

---

## 📊 Comparison Results

### DAVIS Dataset Performance
- RMSE: ~0.7–0.8 log units
- R²: ~0.65–0.70
- Pearson r: ~0.82–0.85
- Samples: 13,000+

### Lassa Dataset Performance
- RMSE: ~0.85 log units  ← Higher due to smaller dataset
- R²: ~0.58–0.65
- Pearson r: ~0.77–0.80
- Samples: 2,442

**Key Insight:** Smaller datasets (Lassa) need more regularization & careful hyperparameter tuning. The improved version (FINAL_LLM_CODE.ipynb) addresses this.

---

## 🎓 Learning Resources

**Inside Notebooks:**
- Multi-modal embedding fusion
- PyTorch Lightning best practices
- Concept bottleneck models for interpretability
- Drug discovery ML pipelines
- Data preprocessing for molecular data

**References:**
- [Concept Bottleneck Models](https://arxiv.org/abs/2007.04871) - Koh et al., 2020
- [ESM2: Language Models for Proteins](https://www.science.org/doi/10.1126/science.ade2574) - Meta AI
- [SELFormer: SMILES Encoding](https://huggingface.co/nferruz/ProtBERT_SMILES)

---

## 💬 Tips & Tricks

- **GPU Memory Issue?** → Reduce `batch_size` to 16 or 8
- **Slow Training?** → Use smaller ESM2 model (`esm2_t6_8M_UR50D`)
- **Overfitting?** → Increase `dropout` or `concept_weight`
- **Underfitting?** → Decrease `dropout`, increase `num_epochs`
- **Need Explanations?** → Extract concepts from model output
- **Want to Deploy?** → Use `main.py` (cleaned up version)

---

## 📋 File Sizes

```
DAVIS_BINDINGAFF.ipynb      ~2.5 MB
FINAL_LLM_CODE.ipynb        ~1.5 MB
```

Keep notebooks for **reference & reproducibility**. Use `main.py` for **production**.

---

**Last Updated:** July 2024  
**Status:** ✅ Both notebooks tested and validated
