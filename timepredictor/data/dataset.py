import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from configs.default import Config


class CornDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def dl_collate_fn(batch):
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    targets = torch.stack(targets)
    return padded_seqs, targets, lengths


def load_data():
    print("Loading data...")
    df = pd.read_csv(Config.data_path)
    if "is_predictable" in df.columns:
        df = df[df["is_predictable"] == 1].reset_index(drop=True)
    groups = df["corn_id"].values

    gss1 = GroupShuffleSplit(n_splits=1, train_size=1.0 - Config.test_ratio, random_state=Config.seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=groups))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, train_size=Config.train_ratio_within_trainval, random_state=Config.seed)
    train_groups = train_val_df["corn_id"].values
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_groups))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(train_df[Config.feature_cols])
    return train_df, val_df, test_df, scaler


def prepare_dl_data(train_df, val_df, test_df, scaler):
    def _build(df):
        seqs, tars = [], []
        for _, group in df.groupby("corn_id"):
            group = group.sort_values("time_minutes")
            feats = scaler.transform(group[Config.feature_cols])
            ys = group[Config.target_col].values
            for i in range(len(group)):
                seqs.append(torch.tensor(feats[: i + 1], dtype=torch.float32))
                tars.append(torch.tensor(ys[i], dtype=torch.float32))
        return seqs, tars

    tr_s, tr_t = _build(train_df)
    va_s, va_t = _build(val_df)
    te_s, te_t = _build(test_df)

    tr_loader = DataLoader(CornDataset(tr_s, tr_t), batch_size=Config.batch_size, shuffle=True, collate_fn=dl_collate_fn)
    va_loader = DataLoader(CornDataset(va_s, va_t), batch_size=Config.batch_size, shuffle=False, collate_fn=dl_collate_fn)
    te_loader = DataLoader(CornDataset(te_s, te_t), batch_size=Config.batch_size, shuffle=False, collate_fn=dl_collate_fn)
    return tr_loader, va_loader, te_loader
