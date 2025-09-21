# -*- coding = utf-8 -*-
# @Time :2025/9/5 15:09
# @Author :郎皓东
# @File ：gnn_optimized.py
# @Software:PyCharm
# gnn_optimized.py
import os, math, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, precision_recall_curve

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- Utils ----------------
def one_hot(series, categories):
    series = series.astype(pd.CategoricalDtype(categories=categories))
    idx = series.cat.codes.values
    eye = np.eye(len(categories), dtype=np.float32)
    out = np.zeros((len(series), len(categories)), dtype=np.float32)
    mask = idx >= 0
    out[np.where(mask)[0]] = eye[idx[mask]]
    return out

def encode_stereo(v):
    table = {
        'complicated':[0,0,0,1,0,0],
        'mixed':[0,0,0,0,1,0],
        '-':[0,0,0,0,0,1],
        'R':[0,0,0,0,0,0],
        'RR':[0,0,1,0,0,0],
        'RRR':[0,1,0,0,0,0],
        'RRS':[0,1,1,0,0,0],
        'RS':[1,0,0,0,0,0],
        'RSS':[1,0,1,0,0,0],
        'S':[1,1,0,0,0,0],
    }
    return np.array(table.get(str(v), [0,0,0,0,0,1]), dtype=np.float32)

def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atom_features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetImplicitValence(),
            int(atom.GetIsAromatic()),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.IsInRing()),
            int(atom.GetHybridization()),
            atom.GetTotalNumHs(),
            int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
        ]
        atom_features.append(feat)
    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[s,e],[e,s]]
        edge_attrs += [[bond.GetBondTypeAsDouble(),
                        int(bond.GetIsAromatic()),
                        int(bond.GetIsConjugated())]]*2
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.zeros((0,3), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ---------------- Model ----------------
class GNNModel(nn.Module):
    def __init__(self, node_dim, edge_dim=3, physchem_dim=4, categorical_dim=20, hidden=128, fused=128):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                              train_eps=True, edge_dim=hidden)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                              train_eps=True, edge_dim=hidden)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                              train_eps=True, edge_dim=hidden)
        self.bn1 = nn.BatchNorm1d(hidden); self.bn2 = nn.BatchNorm1d(hidden); self.bn3 = nn.BatchNorm1d(hidden)

        self.physchem = nn.Sequential(nn.Linear(physchem_dim, 64), nn.ReLU(), nn.Dropout(0.1))
        self.categorical = nn.Sequential(nn.Linear(categorical_dim, 64), nn.ReLU(), nn.Dropout(0.1))

        self.proj_x = nn.Linear(hidden, fused)
        self.proj_p = nn.Linear(64, fused)
        self.proj_c = nn.Linear(64, fused)
        self.attn = nn.Sequential(nn.Linear(fused*3, 128), nn.ReLU(), nn.Linear(128, 3))

        self.cls = nn.Sequential(nn.Linear(fused, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, data):
        ei, ea = data.edge_index, data.edge_attr
        if ea is not None and ea.numel() > 0:
            ea = self.edge_mlp(ea)
        else:
            ea = torch.zeros((ei.size(1), self.edge_mlp[0].out_features), device=ei.device)

        x = self.conv1(data.x, ei, ea); x = self.bn1(x); x = F.relu(x)
        x = self.conv2(x, ei, ea);      x = self.bn2(x); x = F.relu(x)
        x = self.conv3(x, ei, ea);      x = self.bn3(x); x = F.relu(x)
        x = global_mean_pool(x, data.batch)

        p = self.physchem(data.physchem.view(-1,4))
        c = self.categorical(data.categorical.view(x.size(0), -1))

        gx, gp, gc = self.proj_x(x), self.proj_p(p), self.proj_c(c)
        w = torch.softmax(self.attn(torch.cat([gx,gp,gc], dim=1)), dim=1).unsqueeze(-1)
        fused = (torch.stack([gx,gp,gc], dim=1) * w).sum(dim=1)
        logit = self.cls(fused).squeeze(-1)
        return logit

# ---------------- Pipeline ----------------
def main(csv_path="pesticide_data.csv", epochs=80, batch_size=64, lr=1e-3, weight_decay=1e-4):
    df = pd.read_csv(csv_path).dropna(subset=["SMILES","half life[day]"]).reset_index(drop=True)
    df["label"] = (df["half life[day]"] > 4).astype(int)

    physchem_cols = ['logKow','MW(g/mol)','Polar Surface Area (A2)','temperature']
    for col in physchem_cols:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].astype(float).fillna(df[col].median())

    # categories from data
    chiral_cats = sorted(set(df.get("chiral center atom", ['-']))) or ['-']
    comp_cats   = sorted(set(df.get("compartment", ['-']))) or ['-']
    plant_cats  = sorted(set(df.get("plant class", ['-']))) or ['-']

    # split first (no leakage)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    train_df, val_df  = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df["label"])

    scaler = StandardScaler().fit(train_df[physchem_cols])
    def make_physchem(dfx): return scaler.transform(dfx[physchem_cols]).astype(np.float32)
    def make_categorical(dfx):
        ch = one_hot(dfx.get("chiral center atom", pd.Series(["-"]*len(dfx))), chiral_cats)
        cp = one_hot(dfx.get("compartment", pd.Series(["-"]*len(dfx))), comp_cats)
        pc = one_hot(dfx.get("plant class", pd.Series(["-"]*len(dfx))), plant_cats)
        st = np.vstack([encode_stereo(v) for v in dfx.get("Stereochemical configuration", ["-"]*len(dfx))])
        return np.concatenate([ch,cp,pc,st], axis=1).astype(np.float32)

    def build_dataset(dfx):
        P = make_physchem(dfx)
        C = make_categorical(dfx)
        Y = dfx["label"].values.astype(np.float32)
        items = []
        for i, row in dfx.iterrows():
            g = smiles_to_graph(row["SMILES"])
            if g is None or g.x.numel() == 0: continue
            d = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr,
                     y=torch.tensor([Y[dfx.index.get_loc(i)]], dtype=torch.float32))
            d.physchem = torch.from_numpy(P[dfx.index.get_loc(i)])
            d.categorical = torch.from_numpy(C[dfx.index.get_loc(i)])
            items.append(d)
        return items

    train_ds, val_ds, test_ds = build_dataset(train_df), build_dataset(val_df), build_dataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_dim = len(chiral_cats)+len(comp_cats)+len(plant_cats)+6
    model = GNNModel(node_dim=10, categorical_dim=cat_dim).to(device)

    # class imbalance -> pos_weight
    def pos_weight_from(loader):
        ys = []
        for b in loader: ys.extend(b.y.view(-1).numpy().tolist())
        ys = np.array(ys); pos = (ys==1).sum(); neg = (ys==0).sum()
        return torch.tensor([neg/max(pos,1)], dtype=torch.float32, device=device)
    pos_weight = pos_weight_from(train_loader)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    def run_epoch(loader, train_mode=True):
        model.train() if train_mode else model.eval()
        total = 0.0; probs_all=[]; y_all=[]
        for batch in loader:
            batch = batch.to(device)
            if train_mode: optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * batch.num_graphs
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_all.append(batch.y.view(-1).cpu().numpy())
        probs = np.concatenate(probs_all); y = np.concatenate(y_all)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, zero_division=0)
        auc = roc_auc_score(y, probs) if len(np.unique(y))>1 else float("nan")
        mcc = matthews_corrcoef(y, preds) if len(np.unique(preds))>1 else 0.0
        return total/len(loader.dataset), acc, f1, auc, mcc, probs, y

    best_score=-1; best_state=None; best_th=0.5; patience=0
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc, tr_f1, tr_auc, tr_mcc, _, _ = run_epoch(train_loader, True)
        va_loss, va_acc, va_f1, va_auc, va_mcc, va_probs, va_y = run_epoch(val_loader, False)

        # threshold tuning on validation (maximize F1)
        prec, rec, ths = precision_recall_curve(va_y, va_probs)
        f1s = 2*prec*rec/(prec+rec+1e-8)
        if len(ths)>0:
            idx = int(np.nanargmax(f1s[:-1]))
            tuned_th = float(ths[idx])
            tuned_preds = (va_probs >= tuned_th).astype(int)
            tuned_f1 = f1_score(va_y, tuned_preds, zero_division=0)
            tuned_acc = accuracy_score(va_y, tuned_preds)
            tuned_mcc = matthews_corrcoef(va_y, tuned_preds)
            combined = (tuned_f1 + tuned_acc)/2.0
        else:
            tuned_th, tuned_f1, tuned_acc, tuned_mcc = 0.5, va_f1, va_acc, va_mcc
            combined = (va_f1 + va_acc)/2.0

        monitor = va_auc if not math.isnan(va_auc) else combined
        scheduler.step(monitor)

        if combined > best_score:
            best_score = combined; best_th = tuned_th
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch==1 or epoch%10==0:
            print(f"Epoch {epoch:03d} | Train A/F1: {tr_acc:.3f}/{tr_f1:.3f} | "
                  f"Val A/F1: {va_acc:.3f}/{va_f1:.3f} | Tuned(F1/A/M/TH): {tuned_f1:.3f}/{tuned_acc:.3f}/{tuned_mcc:.3f}/{best_th:.2f} | AUC: {va_auc:.3f}")
        if patience>=10: break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k,v in best_state.items()})

    def evaluate(loader, th):
        model.eval()
        probs_all=[]; y_all=[]
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(batch)).cpu().numpy()
            probs_all.append(p); y_all.append(batch.y.view(-1).cpu().numpy())
        probs = np.concatenate(probs_all); y = np.concatenate(y_all)
        preds = (probs >= th).astype(int)
        return dict(
            acc=accuracy_score(y, preds),
            f1=f1_score(y, preds, zero_division=0),
            auc=roc_auc_score(y, probs) if len(np.unique(y))>1 else float("nan"),
            mcc=matthews_corrcoef(y, preds) if len(np.unique(preds))>1 else 0.0,
            threshold=th,
            support=len(y)
        )

    val_metrics = evaluate(val_loader, best_th)
    test_metrics = evaluate(test_loader, best_th)
    print("\n[Validation]", val_metrics)
    print("[Test]      ", test_metrics)

    def shap_analysis(model, df, physchem_cols, chiral_cats, comp_cats, plant_cats):
        # 构造 tabular 特征（理化 + 类别 one-hot + 立体配置）
        scaler = StandardScaler().fit(df[physchem_cols])
        physchem = scaler.transform(df[physchem_cols]).astype(np.float32)

        def one_hot(series, categories):
            series = series.astype(pd.CategoricalDtype(categories=categories))
            idx = series.cat.codes.values
            eye = np.eye(len(categories), dtype=np.float32)
            out = np.zeros((len(series), len(categories)), dtype=np.float32)
            mask = idx >= 0
            out[np.where(mask)[0]] = eye[idx[mask]]
            return out

        ch = one_hot(df.get("chiral center atom", pd.Series(["-"]*len(df))), chiral_cats)
        cp = one_hot(df.get("compartment", pd.Series(["-"]*len(df))), comp_cats)
        pc = one_hot(df.get("plant class", pd.Series(["-"]*len(df))), plant_cats)

        def encode_stereo(v):
            table = {
                'complicated':[0,0,0,1,0,0],
                'mixed':[0,0,0,0,1,0],
                '-':[0,0,0,0,0,1],
                'R':[0,0,0,0,0,0],
                'RR':[0,0,1,0,0,0],
                'RRR':[0,1,0,0,0,0],
                'RRS':[0,1,1,0,0,0],
                'RS':[1,0,0,0,0,0],
                'RSS':[1,0,1,0,0,0],
                'S':[1,1,0,0,0,0],
            }
            return table.get(str(v), [0,0,0,0,0,1])

        st = np.vstack([np.array(encode_stereo(v), dtype=np.float32) for v in df.get("Stereochemical configuration", ["-"]*len(df))])
        cat = np.concatenate([ch,cp,pc,st], axis=1).astype(np.float32)

        X = np.concatenate([physchem, cat], axis=1)

        # Wrapper: 只分析 physchem+categorical
        class TabularWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.physchem = model.physchem
                self.categorical = model.categorical
                self.proj_p = model.proj_p
                self.proj_c = model.proj_c
                self.cls = model.cls
                self.fused = model.proj_p.out_features
            def forward(self, x):
                phys = x[:,:len(physchem_cols)]
                cat = x[:,len(physchem_cols):]
                p = self.physchem(phys)
                c = self.categorical(cat)
                gp, gc = self.proj_p(p), self.proj_c(c)
                fused = 0.5*(gp+gc)  # 简单融合
                return self.cls(fused).squeeze(-1)

        wrapper = TabularWrapper(model).cpu().eval()

        # 1) 定义 numpy 版预测函数：接收 numpy，内部转 torch，返回 numpy
        def predict_fn_np(X_np: np.ndarray) -> np.ndarray:
            model_device = next(wrapper.parameters()).device if any(
                p.requires_grad for p in wrapper.parameters()) else "cpu"
            with torch.no_grad():
                x = torch.from_numpy(X_np).float().to(model_device)
                logits = wrapper(x)  # 形状 [N] 或 [N,1]
                probs = torch.sigmoid(logits).view(-1)  # 用概率更直观，也便于解释
                return probs.cpu().numpy()

        # 2) 用 numpy 做背景数据与解释数据
        explainer = shap.Explainer(predict_fn_np, X)  # X 是 numpy 数组
        shap_values = explainer(X)  # 返回 shap.Explanation
        stereo_map = {
            0: "RS/RSS/S",
            1: "RRR/RRS/S",
            2: "RR/RRS/RSS",
            3: "complicated",
            4: "mixed",
            5: "no_stereo(-)"
        }
        feature_names = physchem_cols + \
            [f"chiral_{c}" for c in chiral_cats] + \
            [f"comp_{c}" for c in comp_cats] + \
            [f"plant_{c}" for c in plant_cats] + \
            [f"stereo_{stereo_map[i]}" for i in range(6)]

        shap.summary_plot(shap_values.values, features=X, feature_names=feature_names, plot_type="violin")
        plt.savefig("shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print("SHAP Beeswarm 图已保存为 shap_beeswarm.png")

    shap_analysis(model, df, physchem_cols, chiral_cats, comp_cats, plant_cats)

if __name__ == "__main__":
    # 默认路径改成你的数据位置
    # main(csv_path="/mnt/data/pesticide_data.csv")
    main(csv_path="data/pesticide_data.csv")

