import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from src.model import AE

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    x_train = np.load("data/processed/x_train.npy").astype(np.float32)
    x_test = np.load("data/processed/x_test.npy").astype(np.float32)
    y_test = np.load("data/processed/y_test.npy").astype(int)


    y_test_bin = (y_test != 1).astype(int)

    train_ds = TensorDataset(torch.from_numpy(x_train))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)

    model = AE(input_dim=x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    os.makedirs("models", exist_ok=True)
    best_path = os.path.join("models", "best_ae.pt")

    best_loss = float("inf")

    for epoch in range(1, 21):
        model.train()
        losses = []

        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        print(f"Epoch {epoch:02d} | train_mse={train_loss:.6f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")


    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    with torch.no_grad():
        xt = torch.from_numpy(x_test).to(device)
        recon = model(xt).cpu().numpy()

    rec_err = ((x_test - recon) ** 2).mean(axis=1)  # per-sample MSE


    with torch.no_grad():
        xtr = torch.from_numpy(x_train).to(device)
        recon_tr = model(xtr).cpu().numpy()
    rec_err_tr = ((x_train - recon_tr) ** 2).mean(axis=1)
    thr = np.percentile(rec_err_tr, 95)

    y_pred = (rec_err >= thr).astype(int)

    auc = roc_auc_score(y_test_bin, rec_err) if len(np.unique(y_test_bin)) > 1 else float("nan")
    f1 = f1_score(y_test_bin, y_pred, zero_division=0)
    cm = confusion_matrix(y_test_bin, y_pred)

    print("\n=== TEST (Anomaly Detection) ===")
    print("AUC (score=rec_error):", round(auc, 4))
    print("F1 (threshold=95p train):", round(f1, 4))
    print("Threshold:", thr)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
