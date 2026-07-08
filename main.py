"""
Lassa Virus Binding Affinity Prediction using Concept-Bottleneck Model (CBM)

A deep learning project combining SELFormer, ESM2, and Morgan fingerprints 
to predict drug-protein binding affinity (pIC50) with interpretability.

Author: Your Name
Date: 2024
"""

import os
import json
import logging
import tqdm
import warnings
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, AllChem, rdMolDescriptors
from rdkit import DataStructs
from transformers import AutoTokenizer, AutoModel
import selfies as sf

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | PyTorch: {torch.__version__}")


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Hyperparameters and paths for training"""
    # Paths
    csv_path: str = 'data/lassa_clean.csv'
    output_dir: str = './outputs'
    model_dir: str = './models'
    embed_dir: str = './embeddings'
    
    # Embedding dimensions
    drug_dim: int = 768          # SELFormer
    prot_dim: int = 320          # ESM2
    morgan_dim: int = 1024       # Morgan fingerprints
    fused_dim: int = 256         # After PCA
    
    # Concept layer
    num_concepts: int = 12
    concept_weight: float = 0.3
    
    # Model architecture
    dropout: float = 0.25
    hidden_dim: int = 512
    num_layers: int = 2
    
    # Training
    batch_size: int = 32
    num_epochs: int = 150
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Data splits
    test_split: float = 0.15
    val_split: float = 0.10
    random_state: int = 42
    
    # Model IDs
    selformer_id: str = 'nferruz/ProtBERT_SMILES'
    esm_id: str = 'facebook/esm2_t6_8M_UR50D'
    
    def __post_init__(self):
        for d in [self.output_dir, self.model_dir, self.embed_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


# ============================================================================
# 2. MOLECULAR DESCRIPTORS & CONCEPTS
# ============================================================================

def calc_concepts(smiles: str, seq: str) -> np.ndarray:
    """
    Calculate 12 interpretable molecular/protein concepts
    
    Concepts:
    0. qed - drug-likeness (Quantitative Estimate of Drug-likeness)
    1. tpsa_norm - normalized topological polar surface area
    2. logp - lipophilicity (partition coefficient)
    3. rotatable_bonds - flexibility
    4. hbd - hydrogen bond donors
    5. hba - hydrogen bond acceptors
    6. mw_norm - normalized molecular weight
    7. aromatic_rings - aromatic content
    8. zinc_chelator - zinc chelation potential
    9. frac_csp3 - fraction of sp3 carbons
    10. seq_length_norm - normalized protein sequence length
    11. binding_compatibility - predicted drug-target interaction potential
    """
    concepts = []
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(12, dtype=np.float32)
        
        # Drug concepts (0-9)
        concepts.append(float(QED.qed(mol)))                    # 0. QED
        concepts.append(float(Descriptors.TPSA(mol)) / 140.0)   # 1. TPSA normalized
        concepts.append(float(Descriptors.MolLogP(mol)))        # 2. LogP
        concepts.append(float(Descriptors.NumRotatableBonds(mol)))  # 3. Rotatable bonds
        concepts.append(float(Descriptors.NumHDonors(mol)))     # 4. H donors
        concepts.append(float(Descriptors.NumHAcceptors(mol)))  # 5. H acceptors
        concepts.append(float(Descriptors.ExactMolWt(mol)) / 500.0)  # 6. MW normalized
        concepts.append(float(Descriptors.NumAromaticRings(mol)))    # 7. Aromatic rings
        
        # Zinc chelation potential (histidine, aspartate, cysteine)
        n_cha = seq.count('H') + seq.count('D') + seq.count('C')
        concepts.append(float(n_cha) / max(1, len(seq)))        # 8. Chelation
        
        # Fraction of sp3 carbons
        frac_csp3 = rdMolDescriptors.CalcFractionCsp3(mol)
        concepts.append(float(frac_csp3))                       # 9. Frac CSP3
        
    except Exception as e:
        logger.warning(f"Error processing SMILES '{smiles}': {e}")
        return np.zeros(12, dtype=np.float32)
    
    # Protein concepts (10-11)
    concepts.append(float(len(seq)) / 500.0)                   # 10. Seq length normalized
    
    # Binding compatibility proxy
    prot_score = min(1.0, float(len(seq)) / 300.0)
    concepts.append(float(prot_score))                         # 11. Binding compatibility
    
    return np.array(concepts, dtype=np.float32)


# ============================================================================
# 3. EMBEDDING ENCODERS
# ============================================================================

class DrugEncoder(nn.Module):
    """SELFormer-based drug (SMILES) encoder"""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.selformer_id)
        self.encoder = AutoModel.from_pretrained(cfg.selformer_id)
        self.norm = nn.LayerNorm(cfg.drug_dim)
        
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        """Encode SMILES strings to dense vectors"""
        tokens = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = self.encoder(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return self.norm(embeddings)


class ProteinEncoder(nn.Module):
    """ESM2 protein sequence encoder"""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.esm_id)
        self.encoder = AutoModel.from_pretrained(cfg.esm_id)
        self.norm = nn.LayerNorm(cfg.prot_dim)
        
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Encode protein sequences to dense vectors"""
        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = self.encoder(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return self.norm(embeddings)


# ============================================================================
# 4. CONCEPT-BOTTLENECK MODEL
# ============================================================================

class ImprovedCBMModel(nn.Module):
    """
    Concept Bottleneck Model for binding affinity prediction
    
    Architecture:
    - Fused embeddings (256D) → Concept extractor → 12 concepts
    - Parallel: Concepts → CBM predictor
    - Parallel: Fused embeddings → Direct bypass predictor
    - Mixture of experts: weighted combination of both predictions
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        # Concept extractor
        self.concept_extractor = nn.Sequential(
            nn.Linear(cfg.fused_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_concepts)
        )
        
        # CBM predictor (concepts → pIC50)
        self.cbm_head = nn.Sequential(
            nn.Linear(cfg.num_concepts, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Bypass predictor (fused embeddings → pIC50 directly)
        self.bypass_head = nn.Sequential(
            nn.Linear(cfg.fused_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(cfg.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Mixture weight
        self.mix_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, fused_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            fused_emb: (batch_size, fused_dim) fused embeddings
            
        Returns:
            pred: (batch_size, 1) predicted pIC50
            concepts: (batch_size, num_concepts) extracted concepts
        """
        # Extract concepts
        concepts = self.concept_extractor(fused_emb)
        
        # CBM path
        cbm_pred = self.cbm_head(concepts)
        
        # Bypass path
        bypass_pred = self.bypass_head(fused_emb)
        
        # Mix predictions
        alpha = torch.sigmoid(self.mix_weight)
        pred = alpha * cbm_pred + (1 - alpha) * bypass_pred
        
        return pred, concepts


# ============================================================================
# 5. DATASET & DATALOADER
# ============================================================================

class LassaDataset(Dataset):
    """PyTorch Dataset for Lassa binding affinity"""
    
    def __init__(self, df: pd.DataFrame, fused_embs: np.ndarray, cfg: Config):
        self.df = df.reset_index(drop=True)
        self.fused_embs = torch.from_numpy(fused_embs).float()
        self.cfg = cfg
        self.orig_idx = df.index.tolist()
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        pic50 = torch.tensor(row['pIC50_norm'], dtype=torch.float32)
        fused_emb = self.fused_embs[idx]
        return fused_emb, pic50.unsqueeze(0)


# ============================================================================
# 6. TRAINING & EVALUATION
# ============================================================================

class ImprovedTrainer:
    """Training loop manager"""
    
    def __init__(self, model: nn.Module, cfg: Config):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Optimizer with 3-level learning rates
        self.optimizer = AdamW([
            {'params': model.concept_extractor.parameters(), 'lr': cfg.learning_rate},
            {'params': model.cbm_head.parameters(), 'lr': cfg.learning_rate},
            {'params': model.bypass_head.parameters(), 'lr': cfg.learning_rate * 3},
            {'params': model.mix_weight, 'lr': cfg.learning_rate}
        ], weight_decay=cfg.weight_decay)
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6
        )
        
        # History
        self.history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
        self.best_r2 = -np.inf
        self.best_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for fused_emb, pic50_true in train_loader:
            fused_emb = fused_emb.to(self.device)
            pic50_true = pic50_true.to(self.device)
            
            # Forward pass
            pic50_pred, concepts = self.model(fused_emb)
            
            # Loss = prediction loss + concept regularization
            pred_loss = self.loss_fn(pic50_pred, pic50_true)
            concept_loss = 0.1 * torch.mean(torch.abs(concepts))
            loss = pred_loss + self.cfg.concept_weight * concept_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(fused_emb)
        
        return total_loss / len(train_loader.dataset)
    
    def evaluate(self, data_loader: DataLoader) -> dict:
        """Evaluate on validation/test set"""
        self.model.eval()
        all_preds, all_trues, all_concepts = [], [], []
        
        with torch.no_grad():
            for fused_emb, pic50_true in data_loader:
                fused_emb = fused_emb.to(self.device)
                pic50_pred, concepts = self.model(fused_emb)
                
                all_preds.append(pic50_pred.cpu().numpy())
                all_trues.append(pic50_true.cpu().numpy())
                all_concepts.append(concepts.cpu().numpy())
        
        preds = np.vstack(all_preds).flatten()
        trues = np.vstack(all_trues).flatten()
        concepts_arr = np.vstack(all_concepts)
        
        mse = mean_squared_error(trues, preds)
        r2 = r2_score(trues, preds)
        pearson_r, _ = pearsonr(trues, preds)
        spearman_r, _ = spearmanr(trues, preds)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'pearson': pearson_r,
            'spearman': spearman_r,
            'preds': preds,
            'trues': trues,
            'concepts': concepts_arr
        }


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def main():
    """Complete training pipeline"""
    
    cfg = Config()
    print("\n" + "="*70)
    print("  LASSA BINDING AFFINITY — IMPROVED CBM MODEL")
    print("="*70 + "\n")
    
    # Load data
    print("📂 Loading dataset...")
    df = pd.read_csv(cfg.csv_path)
    assert all(col in df.columns for col in ['pIC50', 'Smiles', 'sequence'])
    
    # Filter
    df = df[df['pIC50'].between(4.0, 11.0)].dropna(subset=['pIC50', 'Smiles', 'sequence']).copy()
    df = df.reset_index(drop=True)
    print(f"   ✅ Loaded {len(df)} samples")
    print(f"   pIC50: mean={df['pIC50'].mean():.3f}, std={df['pIC50'].std():.3f}")
    print(f"   pIC50 range: {df['pIC50'].min():.2f} – {df['pIC50'].max():.2f}\n")
    
    # Calculate concepts
    print("🧪 Computing molecular concepts...")
    concepts_list = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc='Concepts'):
        c = calc_concepts(row['Smiles'], row['sequence'])
        concepts_list.append(c)
    concepts_array = np.array(concepts_list, dtype=np.float32)
    print(f"   ✅ Computed {concepts_array.shape}\n")
    
    # Train/val/test split
    print("🔀 Splitting data...")
    tv_df, test_df = train_test_split(df, test_size=cfg.test_split, random_state=cfg.random_state)
    val_sz = cfg.val_split / (1 - cfg.test_split)
    train_df, val_df = train_test_split(tv_df, test_size=val_sz, random_state=cfg.random_state)
    
    print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")
    
    # Normalize pIC50
    pic50_mean = float(train_df['pIC50'].mean())
    pic50_std = float(train_df['pIC50'].std())
    
    for split_df in [train_df, val_df, test_df]:
        split_df['pIC50_norm'] = (split_df['pIC50'] - pic50_mean) / pic50_std
    
    # Load or compute embeddings
    print("🔄 Loading embeddings...")
    drug_emb_f = Path(cfg.embed_dir) / 'drug_embs.npy'
    prot_emb_f = Path(cfg.embed_dir) / 'prot_embs.npy'
    morgan_emb_f = Path(cfg.embed_dir) / 'morgan_embs.npy'
    
    if drug_emb_f.exists() and prot_emb_f.exists():
        drug_embs = np.load(drug_emb_f)
        prot_embs = np.load(prot_emb_f)
        morgan_embs = np.load(morgan_emb_f)
        print(f"   ✅ Loaded cached embeddings")
    else:
        print("   ⚠️  Embeddings not found. Compute them separately using:\n"
              "      from main import DrugEncoder, ProteinEncoder\n"
              "      # ... then save with np.save()")
        raise FileNotFoundError("Embeddings not found in ./embeddings/")
    
    # Fuse embeddings
    print("\n🔗 Fusing embeddings...")
    raw_combined = np.hstack([drug_embs, prot_embs, morgan_embs])
    print(f"   Raw combined shape: {raw_combined.shape}")
    
    pca_f = Path(cfg.embed_dir) / 'pca_model.pkl'
    if pca_f.exists():
        with open(pca_f, 'rb') as f:
            pca = pickle.load(f)
        fused_embs = pca.transform(raw_combined)
        print(f"   ✅ Applied cached PCA → {fused_embs.shape}\n")
    else:
        print("   ⚠️  PCA model not found. Training on training data...")
        pca = PCA(n_components=cfg.fused_dim)
        train_idx = train_df.index.tolist()
        pca.fit(raw_combined[train_idx])
        fused_embs = pca.transform(raw_combined)
        
        with open(pca_f, 'wb') as f:
            pickle.dump(pca, f)
        print(f"   ✅ Trained PCA → {fused_embs.shape}\n")
    
    # Create datasets
    train_dataset = LassaDataset(train_df, fused_embs[train_df.index], cfg)
    val_dataset = LassaDataset(val_df, fused_embs[val_df.index], cfg)
    test_dataset = LassaDataset(test_df, fused_embs[test_df.index], cfg)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)
    
    # Model & trainer
    print("🧠 Building model...")
    model = ImprovedCBMModel(cfg).to(device)
    trainer = ImprovedTrainer(model, cfg)
    print(f"   ✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Training loop
    print("🚀 Starting training...")
    best_model_path = Path(cfg.model_dir) / 'best_model.pt'
    
    for epoch in range(cfg.num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_result = trainer.evaluate(val_loader)
        
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_loss'].append(val_result['mse'])
        trainer.history['val_r2'].append(val_result['r2'])
        
        if val_result['r2'] > trainer.best_r2:
            trainer.best_r2 = val_result['r2']
            trainer.best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Val R²: {val_result['r2']:.4f} | Best: {trainer.best_r2:.4f}")
        
        trainer.scheduler.step()
    
    print(f"\n✅ Training complete! Best R² = {trainer.best_r2:.4f} at epoch {trainer.best_epoch}\n")
    
    # Evaluate on test set
    print("📊 Evaluating on test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_result = trainer.evaluate(test_loader)
    
    print("\n" + "="*70)
    print("  TEST SET RESULTS")
    print("="*70)
    print(f"  Samples:         {len(test_df)}")
    print(f"  RMSE:            {test_result['rmse']:.4f}")
    print(f"  R²:              {test_result['r2']:.4f}")
    print(f"  Pearson r:       {test_result['pearson']:.4f}")
    print(f"  Spearman ρ:      {test_result['spearman']:.4f}")
    print("="*70 + "\n")
    
    # Save results
    results = {
        'model_type': 'ImprovedCBM',
        'test_rmse': float(test_result['rmse']),
        'test_r2': float(test_result['r2']),
        'test_pearson': float(test_result['pearson']),
        'test_spearman': float(test_result['spearman']),
        'num_test_samples': len(test_df),
        'config': {
            'batch_size': cfg.batch_size,
            'num_epochs': cfg.num_epochs,
            'learning_rate': cfg.learning_rate,
            'num_concepts': cfg.num_concepts
        }
    }
    
    with open(Path(cfg.output_dir) / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to {cfg.output_dir}/results.json")
    print(f"🧠 Model saved to {best_model_path}")
    
    return model, trainer, test_result


if __name__ == '__main__':
    model, trainer, results = main()
