# SKEMPI Dual-Protein Cross-Attention Model — Task 2 (Mutation Resistance)
#
# Key design decisions:
#   1. Siamese wild-type vs. mutant encoding — the model sees both the original
#      and mutated complex, and the *difference* between their embeddings is
#      what actually represents "what the mutation changed."
#   2. Local mutation-site pooling — global mean-pooling over an entire chain
#      dilutes a single-residue change across hundreds of positions. We also
#      extract the token embedding right at the mutated residue.
#   3. Multi-task head — jointly predicts ddG (regression) and a binary
#      resistant/non-resistant label (classification).

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, f1_score, accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '/content/drive/MyDrive/drug_discovery_project'
CKPT_DIR = f'{DATA_DIR}/checkpoints'

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = 'facebook/esm2_t12_35M_UR50D'
MAX_LEN = 512
MAX_MUTATIONS = 6        # local-pooling slots per complex
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
HEAD_LR = 3e-4            # newly-initialized layers: cross-attn, pooling, head
ENCODER_LR = 2e-5         # pretrained ESM-2 layers: kept conservative
WEIGHT_DECAY = 0.01
EPOCHS = 20
PATIENCE = 5
FREEZE_UP_TO_LAYER = 10   # freeze embeddings + layers 0-9 of 12 (small dataset -> heavy freezing)
DROPOUT = 0.3
N_ATTN_HEADS = 4

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train_df = pd.read_csv(f'{DATA_DIR}/skempi_train_final.csv')
val_df = pd.read_csv(f'{DATA_DIR}/skempi_val_final.csv')
test_df = pd.read_csv(f'{DATA_DIR}/skempi_test_final.csv')

TRAIN_DDG_MEAN = train_df['ddG_kcal_mol'].mean()
TRAIN_DDG_STD = train_df['ddG_kcal_mol'].std()

# ---------------------------------------------------------------------------
# Dataset — parses mutation strings into (chain, position) so the model can
# later pull out the token embedding at the exact mutated residue.
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MUT_RE = re.compile(r'^([A-Za-z])([A-Za-z])(\d+)([A-Za-z])$')  # e.g. "LI45G" -> wt=L chain=I pos=45 mut=G


def parse_mutation_positions(mut_str, chains_side1, chains_side2, max_len):
    pos_a, mask_a = [0] * MAX_MUTATIONS, [0.0] * MAX_MUTATIONS
    pos_b, mask_b = [0] * MAX_MUTATIONS, [0.0] * MAX_MUTATIONS
    ia, ib = 0, 0
    for m in str(mut_str).split(','):
        match = MUT_RE.match(m.strip())
        if not match:
            continue
        _, chain, pos_str, _ = match.groups()
        pos = int(pos_str)  # ESM-2 prepends <cls>, so 1-indexed AA position = token index
        if pos < 1 or pos > max_len - 2:
            continue
        if chain in str(chains_side1) and ia < MAX_MUTATIONS:
            pos_a[ia], mask_a[ia] = pos, 1.0
            ia += 1
        elif chain in str(chains_side2) and ib < MAX_MUTATIONS:
            pos_b[ib], mask_b[ib] = pos, 1.0
            ib += 1
    return pos_a, mask_a, pos_b, mask_b


class SkempiDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN, ddg_mean=TRAIN_DDG_MEAN, ddg_std=TRAIN_DDG_STD):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ddg_mean, self.ddg_std = ddg_mean, ddg_std

    def __len__(self):
        return len(self.df)

    def _tok(self, seq):
        enc = self.tokenizer(str(seq) if pd.notna(seq) else '', truncation=True,
                              max_length=self.max_len, padding='max_length', return_tensors='pt')
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a_wt_ids, a_wt_mask = self._tok(row['seq1_wt'])
        b_wt_ids, b_wt_mask = self._tok(row['seq2_wt'])
        a_mut_ids, a_mut_mask = self._tok(row['seq1_mut'])
        b_mut_ids, b_mut_mask = self._tok(row['seq2_mut'])
        pos_a, mask_a, pos_b, mask_b = parse_mutation_positions(
            row['Mutation(s)_cleaned'], row['chains_side1'], row['chains_side2'], self.max_len)

        return {
            'a_wt_ids': a_wt_ids, 'a_wt_mask': a_wt_mask, 'b_wt_ids': b_wt_ids, 'b_wt_mask': b_wt_mask,
            'a_mut_ids': a_mut_ids, 'a_mut_mask': a_mut_mask, 'b_mut_ids': b_mut_ids, 'b_mut_mask': b_mut_mask,
            'pos_a': torch.tensor(pos_a), 'mask_a': torch.tensor(mask_a, dtype=torch.float32),
            'pos_b': torch.tensor(pos_b), 'mask_b': torch.tensor(mask_b, dtype=torch.float32),
            'ddg_norm': torch.tensor((row['ddG_kcal_mol'] - self.ddg_mean) / self.ddg_std, dtype=torch.float32),
            'ddg_raw': torch.tensor(row['ddG_kcal_mol'], dtype=torch.float32),
            'label': torch.tensor(row['resistant_label'], dtype=torch.float32),
        }


train_loader = DataLoader(SkempiDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SkempiDataset(val_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(SkempiDataset(test_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """Query sequence attends over Key/Value sequence (padding-masked)."""
    def __init__(self, hidden_dim, n_heads=N_ATTN_HEADS, dropout=DROPOUT):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, kv, kv_mask):
        attn_out, _ = self.attn(query, kv, kv, key_padding_mask=(kv_mask == 0))
        return self.norm(query + attn_out)


def masked_mean_pool(x, mask):
    mask = mask.unsqueeze(-1).float()
    return (x * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


def gather_local_mean(x, positions, valid_mask):
    """Mean of token embeddings at the mutated residue position(s)."""
    idx = positions.clamp(0, x.shape[1] - 1).unsqueeze(-1).expand(-1, -1, x.shape[-1])
    gathered = torch.gather(x, 1, idx)
    mask = valid_mask.unsqueeze(-1)
    return (gathered * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


class SkempiCrossAttentionModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, freeze_up_to_layer=FREEZE_UP_TO_LAYER, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < freeze_up_to_layer:
                for p in layer.parameters():
                    p.requires_grad = False

        self.cross_a_to_b = CrossAttentionBlock(hidden_dim)
        self.cross_b_to_a = CrossAttentionBlock(hidden_dim)

        combined_dim = hidden_dim * 4 * 3  # [global+local] x [wt, mut, diff]
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.ddg_head = nn.Linear(64, 1)
        self.cls_head = nn.Linear(64, 1)

    def encode_complex(self, a_ids, a_mask, b_ids, b_mask, pos_a, mask_a, pos_b, mask_b):
        a_tok = self.encoder(input_ids=a_ids, attention_mask=a_mask).last_hidden_state
        b_tok = self.encoder(input_ids=b_ids, attention_mask=b_mask).last_hidden_state
        fused_a = self.cross_a_to_b(a_tok, b_tok, b_mask)
        fused_b = self.cross_b_to_a(b_tok, a_tok, a_mask)

        global_vec = torch.cat([masked_mean_pool(fused_a, a_mask), masked_mean_pool(fused_b, b_mask)], -1)
        local_vec = torch.cat([gather_local_mean(fused_a, pos_a, mask_a),
                                gather_local_mean(fused_b, pos_b, mask_b)], -1)
        return torch.cat([global_vec, local_vec], -1)

    def forward(self, batch):
        wt = self.encode_complex(batch['a_wt_ids'], batch['a_wt_mask'], batch['b_wt_ids'], batch['b_wt_mask'],
                                  batch['pos_a'], batch['mask_a'], batch['pos_b'], batch['mask_b'])
        mut = self.encode_complex(batch['a_mut_ids'], batch['a_mut_mask'], batch['b_mut_ids'], batch['b_mut_mask'],
                                   batch['pos_a'], batch['mask_a'], batch['pos_b'], batch['mask_b'])
        features = self.head(torch.cat([wt, mut, mut - wt], -1))
        return self.ddg_head(features).squeeze(-1), self.cls_head(features).squeeze(-1)


model = SkempiCrossAttentionModel().to(DEVICE)

# ---------------------------------------------------------------------------
# Training — differential learning rates: pretrained ESM-2 layers get a
# lower LR than the freshly-initialized cross-attention/pooling/head layers.
# ---------------------------------------------------------------------------
encoder_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')]

optimizer = torch.optim.AdamW(
    [{'params': encoder_params, 'lr': ENCODER_LR}, {'params': head_params, 'lr': HEAD_LR}],
    weight_decay=WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(0.1 * len(train_loader) // GRAD_ACCUM_STEPS * EPOCHS),
    len(train_loader) // GRAD_ACCUM_STEPS * EPOCHS)

mse_loss, bce_loss = nn.MSELoss(), nn.BCEWithLogitsLoss()


def move_batch(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def run_epoch(loader, train):
    model.train() if train else model.eval()
    total_loss, optimizer_step = 0.0, 0
    optimizer.zero_grad()
    with torch.set_grad_enabled(train):
        for step, batch in enumerate(loader):
            batch = move_batch(batch)
            ddg_pred, cls_logit = model(batch)
            loss = mse_loss(ddg_pred, batch['ddg_norm']) + bce_loss(cls_logit, batch['label'])
            if train:
                (loss / GRAD_ACCUM_STEPS).backward()
                if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
    return total_loss / len(loader)


best_val_loss, patience_counter = float('inf'), 0
history = {'train_loss': [], 'val_loss': []}

for epoch in range(EPOCHS):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} — train: {train_loss:.4f}  val: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss, patience_counter = val_loss, 0
        torch.save(model.state_dict(), f'{CKPT_DIR}/skempi_crossattn_v2_best.pt')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(loader, name):
    model.load_state_dict(torch.load(f'{CKPT_DIR}/skempi_crossattn_v2_best.pt'))
    model.eval()
    ddg_true, ddg_pred, cls_true, cls_prob = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out_ddg, out_cls = model(move_batch(batch))
            ddg_true.extend(batch['ddg_raw'].numpy())
            ddg_pred.extend(out_ddg.cpu().numpy() * TRAIN_DDG_STD + TRAIN_DDG_MEAN)
            cls_true.extend(batch['label'].numpy())
            cls_prob.extend(torch.sigmoid(out_cls).cpu().numpy())

    ddg_true, ddg_pred = np.array(ddg_true), np.array(ddg_pred)
    cls_true, cls_prob = np.array(cls_true), np.array(cls_prob)
    cls_pred = (cls_prob >= 0.5).astype(int)

    print(f"=== {name} ===")
    print(f"R2: {r2_score(ddg_true, ddg_pred):.4f}  Pearson r: {pearsonr(ddg_true, ddg_pred)[0]:.4f}  "
          f"RMSE: {np.sqrt(mean_squared_error(ddg_true, ddg_pred)):.4f}")
    print(f"AUC: {roc_auc_score(cls_true, cls_prob):.4f}  F1: {f1_score(cls_true, cls_pred):.4f}  "
          f"Accuracy: {accuracy_score(cls_true, cls_pred):.4f}")


evaluate(val_loader, 'Validation')
evaluate(test_loader, 'Test')
