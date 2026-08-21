"""
Cross-Attention Fusion Model — Drug-Target Binding Affinity (pKi) Prediction
BindingDB | ChemBERTa (ligand) + ESM-2 (protein) + Cross-Attention Fusion
Condensed for presentation — see full Colab notebook for caching/resume logic.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Data loading & cleaning
# ---------------------------------------------------------------------------
def clean_smiles(s):
    return str(s).split(' ')[0]  # strip CXSMILES annotation tags

def load_and_clean(path):
    df = pd.read_csv(path).rename(columns={
        'Ligand SMILES': 'smiles',
        'BindingDB Target Chain Sequence 1': 'protein_seq',
        'Target Name': 'target_name'
    })
    df['smiles'] = df['smiles'].apply(clean_smiles)
    df = df[df['smiles'].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
    return df.groupby(['smiles', 'target_name'], as_index=False).agg(
        {'pKi': 'mean', 'protein_seq': 'first'}
    )

df_train_full = load_and_clean("bindingdb_train_final.csv")
df_test = load_and_clean("bindingdb_test_final.csv")

# Scaffold-grouped split — prevents structurally similar molecules leaking across splits
df_train_full['scaffold'] = df_train_full['smiles'].apply(
    lambda s: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)))
)
train_idx, val_idx = next(GroupShuffleSplit(test_size=0.10, random_state=42)
                           .split(df_train_full, groups=df_train_full['scaffold']))
df_train, df_val = df_train_full.iloc[train_idx], df_train_full.iloc[val_idx]

# ---------------------------------------------------------------------------
# 2. Tokenization & Dataset
# ---------------------------------------------------------------------------
LIGAND_MODEL, PROTEIN_MODEL = "seyonec/ChemBERTa-zinc-base-v1", "facebook/esm2_t12_35M_UR50D"
ligand_tok = AutoTokenizer.from_pretrained(LIGAND_MODEL)
protein_tok = AutoTokenizer.from_pretrained(PROTEIN_MODEL)
MAX_LIGAND_LEN, MAX_PROTEIN_LEN = 128, 1024

class AffinityDataset(Dataset):
    def __init__(self, df):
        self.smiles, self.proteins = df['smiles'].tolist(), df['protein_seq'].tolist()
        self.labels = df['pKi'].astype(float).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        l = ligand_tok(self.smiles[idx], max_length=MAX_LIGAND_LEN, padding='max_length',
                        truncation=True, return_tensors='pt')
        p = protein_tok(self.proteins[idx], max_length=MAX_PROTEIN_LEN, padding='max_length',
                         truncation=True, return_tensors='pt')
        return {
            'ligand_input_ids': l['input_ids'].squeeze(0), 'ligand_attention_mask': l['attention_mask'].squeeze(0),
            'protein_input_ids': p['input_ids'].squeeze(0), 'protein_attention_mask': p['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ---------------------------------------------------------------------------
# 3. Cross-Attention Fusion Model
# ---------------------------------------------------------------------------
class CrossAttentionFusionModel(nn.Module):
    def __init__(self, ligand_model, protein_model, fusion_dim=256, n_heads=4,
                 dropout=0.3, n_unfrozen_layers=2):
        super().__init__()
        self.ligand_encoder = AutoModel.from_pretrained(ligand_model)
        self.protein_encoder = AutoModel.from_pretrained(protein_model)

        # Freeze encoders, unfreeze last N layers of each (partial fine-tuning)
        for enc in (self.ligand_encoder, self.protein_encoder):
            for p in enc.parameters():
                p.requires_grad = False
            for layer in enc.encoder.layer[-n_unfrozen_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.ligand_proj = nn.Linear(self.ligand_encoder.config.hidden_size, fusion_dim)
        self.protein_proj = nn.Linear(self.protein_encoder.config.hidden_size, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(fusion_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, l_ids, l_mask, p_ids, p_mask):
        ligand_feat = self.ligand_proj(self.ligand_encoder(l_ids, attention_mask=l_mask).last_hidden_state)
        protein_feat = self.protein_proj(self.protein_encoder(p_ids, attention_mask=p_mask).last_hidden_state)

        # Ligand queries attend over protein residues -> learns binding-relevant substructures
        fused, _ = self.cross_attn(query=ligand_feat, key=protein_feat, value=protein_feat,
                                    key_padding_mask=(p_mask == 0))
        fused = self.dropout(self.norm(fused + ligand_feat))

        mask = l_mask.unsqueeze(-1).float()
        pooled = (fused * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.regression_head(pooled).squeeze(-1)

# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
BATCH_SIZE, MAX_EPOCHS, LR, PATIENCE = 8, 8, 1e-4, 3

train_loader = DataLoader(AffinityDataset(df_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(AffinityDataset(df_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(AffinityDataset(df_test), batch_size=BATCH_SIZE)

model = CrossAttentionFusionModel(LIGAND_MODEL, PROTEIN_MODEL).to(device)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
criterion = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    for batch in loader:
        inputs = [batch[k].to(device) for k in
                  ['ligand_input_ids', 'ligand_attention_mask', 'protein_input_ids', 'protein_attention_mask']]
        labels = batch['label'].to(device)
        with torch.cuda.amp.autocast():
            preds = model(*inputs)
            loss = criterion(preds, labels)
        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(labels)
        preds_all += preds.detach().float().cpu().tolist()
        labels_all += labels.float().cpu().tolist()
    return total_loss / len(loader.dataset), np.array(preds_all), np.array(labels_all)

best_val_loss, epochs_no_improve = float('inf'), 0
for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_preds, val_labels = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss, epochs_no_improve = val_loss, 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# ---------------------------------------------------------------------------
# 5. Final test-set evaluation
# ---------------------------------------------------------------------------
model.load_state_dict(torch.load("best_model.pt"))
_, test_preds, test_labels = run_epoch(test_loader, train=False)

rmse = np.sqrt(np.mean((test_preds - test_labels) ** 2))
mae = np.mean(np.abs(test_preds - test_labels))
r2 = 1 - np.sum((test_labels - test_preds) ** 2) / np.sum((test_labels - test_labels.mean()) ** 2)
pearson_r, _ = pearsonr(test_preds, test_labels)

print(f"Test RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  Pearson r={pearson_r:.4f}")
