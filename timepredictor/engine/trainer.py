import copy
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

from configs.default import Config, set_seed
from data.dataset import load_data, prepare_dl_data
from models.cnn_bilstm_cbam_bra import ImprovedBiLSTM


def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom))


def set_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


set_nature_style()


def save_checkpoint(model, scaler, metrics, timing, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_cols": list(Config.feature_cols),
        "target_col": Config.target_col,
        "metrics": metrics,
        "timing": timing,
        "model_name": "ImprovedBiLSTM",
        "hidden_size": Config.hidden_size,
        "num_layers": Config.num_layers,
        "dropout": Config.dropout,
    }
    torch.save(checkpoint, save_path)

    meta_path = save_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_path": str(save_path),
                "metrics": metrics,
                "timing": timing,
                "feature_dim": len(Config.feature_cols),
                "device": str(Config.device),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


class ResultManager:
    def __init__(self, name, folder_name):
        self.name = name
        self.save_dir = os.path.join(str(Config.output_root), folder_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.excel_writer = pd.ExcelWriter(os.path.join(self.save_dir, f"{name}_Plot_Data.xlsx"), engine="openpyxl")

    def save_plot_data(self, sheet_name, data_dict):
        max_len = max(len(v) if hasattr(v, "__len__") else 1 for v in data_dict.values())
        for k, v in data_dict.items():
            if not hasattr(v, "__len__"):
                data_dict[k] = [v] * max_len
            elif len(v) < max_len:
                data_dict[k] = list(v) + [None] * (max_len - len(v))
        df = pd.DataFrame(data_dict)
        df.to_excel(self.excel_writer, sheet_name=sheet_name[:31], index=False)

    def finalize(self):
        self.excel_writer.close()

    def generate_all_plots(self, trues, preds, history, eval_df, scaler, model, model_type="dl", timing=None):
        residuals = preds - trues
        r2 = r2_score(trues, preds)

        plt.figure(figsize=(8, 6))
        if history:
            x_axis = history.get("x", range(1, len(history["train_loss"]) + 1))
            t_loss = np.array(history["train_loss"]) / 100.0
            v_loss = np.array(history["val_loss"]) / 100.0
            plt.plot(x_axis, t_loss, label="Train", color=Config.colors["red"])
            plt.plot(x_axis, v_loss, label="Val", color=Config.colors["blue"])
            plt.xlabel("Epochs/Iterations")
            plt.ylabel("Scaled Loss")
            self.save_plot_data("Fig1_Learning_Curve", {"X": x_axis, "Train_Loss": t_loss, "Val_Loss": v_loss})
        plt.title("Learning Curve")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "Fig1_Learning_Curve.png"))
        plt.close()

        plt.figure(figsize=(8, 7))
        hb = plt.hexbin(trues, preds, gridsize=30, cmap="Blues", mincnt=1, edgecolors="none")
        plt.plot([0, max(trues)], [0, max(trues)], "r--", linewidth=1.5)
        cb = plt.colorbar(hb)
        cb.set_label("Count")
        plt.title(f"Prediction Density (R^2={r2:.3f})")
        plt.xlabel("True (min)")
        plt.ylabel("Pred (min)")
        plt.savefig(os.path.join(self.save_dir, "Fig2_Prediction_Density.png"))
        plt.close()
        self.save_plot_data("Fig2_Prediction_Density", {"True": trues, "Pred": preds})

        plt.figure(figsize=(8, 6))
        means = (preds + trues) / 2
        plt.scatter(means, residuals, alpha=0.6, s=20, color=Config.colors["dark_blue"], edgecolors="w", linewidth=0.3)
        plt.axhline(np.mean(residuals), color="k")
        plt.axhline(np.mean(residuals) + 1.96 * np.std(residuals), color="r", linestyle="--")
        plt.axhline(np.mean(residuals) - 1.96 * np.std(residuals), color="r", linestyle="--")
        plt.title("Residuals vs Mean")
        plt.savefig(os.path.join(self.save_dir, "Fig3_Bland_Altman.png"))
        plt.close()
        self.save_plot_data("Fig3_Bland_Altman", {"Mean": means, "Diff": residuals})

        plt.figure(figsize=(8, 6))
        residuals = np.asarray(pd.Series(residuals).dropna().to_numpy(), dtype=float).ravel()
        plt.hist(residuals, bins=30, density=True, color=Config.colors["green"], alpha=0.5, edgecolor="white")
        try:
            kde = stats.gaussian_kde(residuals)
            xx = np.linspace(float(np.min(residuals)), float(np.max(residuals)), 200)
            plt.plot(xx, kde(xx), "k--", linewidth=1.5)
        except Exception:
            pass
        plt.title("Error Distribution")
        plt.savefig(os.path.join(self.save_dir, "Fig4_Error_Distribution.png"))
        plt.close()
        self.save_plot_data("Fig4_Error_Distribution", {"Residuals": residuals})

        plt.figure(figsize=(8, 6))
        res = stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot")
        plt.savefig(os.path.join(self.save_dir, "Fig5_QQ_Plot.png"))
        plt.close()
        self.save_plot_data("Fig5_QQ_Plot", {"Theoretical_Quantiles": res[0][0], "Ordered_Values": res[0][1]})

        plt.figure(figsize=(8, 6))
        abs_err = np.abs(residuals)
        sorted_ae = np.sort(abs_err)
        p = 1.0 * np.arange(len(sorted_ae)) / (len(sorted_ae) - 1)
        plt.plot(sorted_ae, p, color=Config.colors["purple"], linewidth=2)
        plt.title("CDF of Absolute Error")
        plt.xlabel("Absolute Error")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "Fig6_CDF_Error.png"))
        plt.close()
        self.save_plot_data("Fig6_CDF_Error", {"Sorted_AE": sorted_ae, "Probability": p})

        plt.figure(figsize=(8, 5))
        sample_pool = eval_df["corn_id"].unique()
        sample_count = min(3, len(sample_pool))
        sample_ids = np.random.choice(sample_pool, sample_count, replace=False)
        plot_data = {}
        colors = [Config.colors["red"], Config.colors["blue"], Config.colors["green"], Config.colors["orange"]]
        for i, cid in enumerate(sample_ids):
            grp = eval_df[eval_df["corn_id"] == cid].sort_values("time_minutes")
            times = grp["time_minutes"].values
            ys = grp[Config.target_col].values
            ft = scaler.transform(grp[Config.feature_cols])
            p_traj = []
            for j in range(len(grp)):
                seq = torch.tensor(ft[: j + 1], dtype=torch.float32).unsqueeze(0).to(Config.device)
                l = torch.tensor([j + 1]).cpu()
                with torch.no_grad():
                    p_traj.append(model(seq, l).item())
            col = colors[i % len(colors)]
            plt.plot(times, ys, "--", color=col, alpha=0.5, label=f"True #{cid}")
            plt.plot(times, p_traj, "o-", color=col, markersize=4, label=f"Pred #{cid}")
            plot_data[f"ID_{cid}_Time"] = times
            plot_data[f"ID_{cid}_True"] = ys
            plot_data[f"ID_{cid}_Pred"] = p_traj
        plt.title("Drying Trajectories")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "Fig7_Drying_Trajectories.png"))
        plt.close()
        if plot_data:
            max_l = max(len(v) for v in plot_data.values())
            for k in plot_data:
                arr = np.array(plot_data[k], dtype=float)
                plot_data[k] = np.pad(arr, (0, max_l - len(arr)), constant_values=np.nan)
            self.save_plot_data("Fig7_Drying_Trajectories", plot_data)

        if timing:
            plt.figure(figsize=(8, 6))
            metrics = ["Train Time (s)", "Inference Time (s)"]
            values = [timing["train_time"], timing["inference_time"]]
            plt.bar(metrics, values, color=[Config.colors["orange"], Config.colors["cyan"]], width=0.5)
            for i, v in enumerate(values):
                plt.text(i, v + 0.01 * max(values), f"{v:.2f}", ha="center", va="bottom")
            plt.title("Time Efficiency")
            plt.ylabel("Time (seconds)")
            plt.savefig(os.path.join(self.save_dir, "Fig8_Time_Efficiency.png"))
            plt.close()
            self.save_plot_data("Fig8_Time_Efficiency", {"Metric": metrics, "Value": values})

        self.finalize()
        self.write_md_report(trues, preds, timing)

    def write_md_report(self, trues, preds, timing=None):
        mae = mean_absolute_error(trues, preds)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        r2 = r2_score(trues, preds)
        mape = safe_mape(trues, preds)

        time_str = ""
        if timing:
            time_str = f"""| **Train Time** | {timing['train_time']:.2f} s |
| **Inference Time** | {timing['inference_time']:.2f} s |
| **Best Epoch** | {timing['best_epoch']} / {timing['max_epochs']} |
| **Stopped Epoch** | {timing['stopped_epoch']} |
| **Best Val Loss** | {timing['best_val_loss']:.4f} |
| **Early Stopping Patience** | {timing['early_stopping_patience']} |
| **Early Stopping Min Delta** | {timing['early_stopping_min_delta']:.6f} |"""

        content = f"""# {self.name} Report
| Metric | Value |
| :--- | :--- |
| **R^2** | {r2:.4f} |
| **MAE** | {mae:.4f} |
| **RMSE** | {rmse:.4f} |
| **MAPE** | {mape:.4%} |
{time_str}
"""
        with open(os.path.join(self.save_dir, "Model_Description_Report.md"), "w", encoding="utf-8") as f:
            f.write(content)


def run_dl(name, model_cls, tr_loader, va_loader, test_loader, test_df, scaler, folder):
    print(f"Running {name}...")
    model = model_cls().to(Config.device)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=Config.lr)
    hist = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    start_train = time.time()
    epoch_bar = tqdm(range(Config.epochs), desc=f"{name} Epochs", dynamic_ncols=True)
    for ep in epoch_bar:
        model.train()
        tl = 0
        train_bar = tqdm(tr_loader, desc=f"Epoch {ep + 1}/{Config.epochs} [Train]", leave=False, dynamic_ncols=True)
        for s, t, l in train_bar:
            s, t = s.to(Config.device), t.to(Config.device)
            l = l.cpu()
            opt.zero_grad()
            out = model(s, l)
            loss = crit(out, t)
            loss.backward()
            opt.step()
            tl += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{tl / (train_bar.n + 1):.4f}")

        model.eval()
        vl = 0
        with torch.no_grad():
            val_bar = tqdm(va_loader, desc=f"Epoch {ep + 1}/{Config.epochs} [Val]", leave=False, dynamic_ncols=True)
            for s, t, l in val_bar:
                s, t = s.to(Config.device), t.to(Config.device)
                l = l.cpu()
                v_loss = crit(model(s, l), t).item()
                vl += v_loss
                val_bar.set_postfix(loss=f"{v_loss:.4f}", avg=f"{vl / (val_bar.n + 1):.4f}")

        train_epoch_loss = tl / len(tr_loader)
        val_epoch_loss = vl / len(va_loader)
        hist["train_loss"].append(train_epoch_loss)
        hist["val_loss"].append(val_epoch_loss)

        if val_epoch_loss < best_val_loss - Config.early_stopping_min_delta:
            best_val_loss = val_epoch_loss
            best_epoch = ep + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_bar.set_postfix(
            train=f"{train_epoch_loss:.4f}",
            val=f"{val_epoch_loss:.4f}",
            best=f"{best_val_loss:.4f}",
            patience=f"{patience_counter}/{Config.early_stopping_patience}",
        )

        if patience_counter >= Config.early_stopping_patience:
            tqdm.write(
                f"{name}: early stopping at epoch {ep + 1}, best epoch {best_epoch}, "
                f"best val loss {best_val_loss:.4f}"
            )
            break
    train_time = time.time() - start_train

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    preds, trues = [], []
    start_infer = time.time()
    with torch.no_grad():
        for s, t, l in test_loader:
            s, t = s.to(Config.device), t.to(Config.device)
            l = l.cpu()
            preds.extend(model(s, l).detach().cpu().numpy())
            trues.extend(t.detach().cpu().numpy())
    inference_time = time.time() - start_infer

    mgr = ResultManager(name, folder)
    metrics = {
        "mse": mean_squared_error(np.array(trues), np.array(preds)),
        "mae": mean_absolute_error(np.array(trues), np.array(preds)),
        "rmse": np.sqrt(mean_squared_error(np.array(trues), np.array(preds))),
        "r2": r2_score(np.array(trues), np.array(preds)),
        "mape": safe_mape(np.array(trues), np.array(preds)),
    }
    timing_payload = {
        "train_time": train_time,
        "inference_time": inference_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopped_epoch": len(hist["train_loss"]),
        "max_epochs": Config.epochs,
        "early_stopping_patience": Config.early_stopping_patience,
        "early_stopping_min_delta": Config.early_stopping_min_delta,
    }
    mgr.generate_all_plots(
        np.array(trues),
        np.array(preds),
        hist,
        test_df,
        scaler,
        model,
        "dl",
        timing_payload,
    )
    save_checkpoint(model, scaler, metrics, timing_payload, Config.checkpoint_path)
    return {
        "name": name,
        "metrics": metrics,
        "checkpoint_path": str(Config.checkpoint_path),
    }


def run_improved_model(seed=None, save_artifacts=True, folder_suffix=""):
    if seed is not None:
        Config.seed = seed
        set_seed(seed)

    train_df, val_df, test_df, scaler = load_data()
    tr_dl, va_dl, te_dl = prepare_dl_data(train_df, val_df, test_df, scaler)
    folder = f"00_Improved_CNN_BiLSTM_CBAM_BRA{folder_suffix}" if folder_suffix else "00_Improved_CNN_BiLSTM_CBAM_BRA"

    if save_artifacts:
        return run_dl("Improved_CNN_BiLSTM_CBAM_BRA", ImprovedBiLSTM, tr_dl, va_dl, te_dl, test_df, scaler, folder)

    model = ImprovedBiLSTM().to(Config.device)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=Config.lr)
    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    for _ in range(Config.epochs):
        model.train()
        for s, t, l in tr_dl:
            s, t = s.to(Config.device), t.to(Config.device)
            opt.zero_grad()
            out = model(s, l.cpu())
            loss = crit(out, t)
            loss.backward()
            opt.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for s, t, l in va_dl:
                s, t = s.to(Config.device), t.to(Config.device)
                vl += crit(model(s, l.cpu()), t).item()
        val_epoch_loss = vl / len(va_dl)
        if val_epoch_loss < best_val_loss - Config.early_stopping_min_delta:
            best_val_loss = val_epoch_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= Config.early_stopping_patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for s, t, l in te_dl:
            s, t = s.to(Config.device), t.to(Config.device)
            preds.extend(model(s, l.cpu()).detach().cpu().numpy())
            trues.extend(t.detach().cpu().numpy())
    trues = np.array(trues)
    preds = np.array(preds)
    return {
        "name": "Improved_CNN_BiLSTM_CBAM_BRA",
        "metrics": {
            "mse": mean_squared_error(trues, preds),
            "mae": mean_absolute_error(trues, preds),
            "rmse": np.sqrt(mean_squared_error(trues, preds)),
            "r2": r2_score(trues, preds),
            "mape": safe_mape(trues, preds),
        },
    }
