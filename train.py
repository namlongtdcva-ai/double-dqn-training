"""
Huấn luyện + đánh giá Double DQN phát hiện tấn công V2X (6 lớp)
"""
import os
import gc
import json
import time
import random

import numpy as np
import pandas as pd
import torch
from sklearn.tree import DecisionTreeClassifier

from preprocessing import CFG, CLASS_NAMES, K, STATE, load_all_splits
from env import TorchBatchEnv, class_weights_from_labels
from model import DoubleDQNAgent


# ĐO LƯỜNG
def conf_matrix(y, p, k=K):
    return np.bincount(y.astype(np.int64) * k + p.astype(np.int64), minlength=k * k).reshape(k, k)


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    pred_n = cm.sum(0).astype(np.float64)
    prec = tp / np.maximum(pred_n, 1)
    rec = tp / np.maximum(support, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    w = support / support.sum()
    return dict(accuracy=float(tp.sum() / cm.sum()),
                macro=dict(precision=float(prec.mean()), recall=float(rec.mean()), f1=float(f1.mean())),
                weighted=dict(precision=float((prec * w).sum()), recall=float((rec * w).sum()),
                              f1=float((f1 * w).sum())),
                per_class={CLASS_NAMES[i]: dict(precision=float(prec[i]), recall=float(rec[i]),
                                                f1=float(f1[i]), support=int(support[i])) for i in range(len(prec))})


def binary_from_cm(cm):
    """Gộp: benign (lớp 0) vs bất kỳ loại tấn công nào."""
    tn = cm[0, 0]; fp = cm[0, 1:].sum(); fn = cm[1:, 0].sum(); tp = cm[1:, 1:].sum()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(precision=float(p), recall=float(r), f1=float(2 * p * r / max(p + r, 1e-12)),
                false_positive_rate=float(fp / max(fp + tn, 1)))


def print_report(title, cm):
    m = metrics_from_cm(cm)
    print(f"\n================ {title} ================")
    print(f"Accuracy {m['accuracy']:.4f} | Macro P {m['macro']['precision']:.4f} | "
          f"Macro R {m['macro']['recall']:.4f} | Macro F1 {m['macro']['f1']:.4f} | "
          f"Weighted F1 {m['weighted']['f1']:.4f}")
    print(f"{'':>20}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>12}")
    for c, v in m["per_class"].items():
        print(f"{c:>20}{v['precision']:>11.4f}{v['recall']:>9.4f}{v['f1']:>9.4f}{v['support']:>12,}")
    b = binary_from_cm(cm)
    print(f"[Gộp benign vs attack] precision {b['precision']:.4f} | recall {b['recall']:.4f} | "
          f"F1 {b['f1']:.4f} | tỉ lệ dương tính giả {b['false_positive_rate']:.4f}")
    print("Confusion matrix (hàng = nhãn thật, cột = dự đoán):")
    print(cm)
    m["binary_attack_vs_benign"] = b
    return m


def calibrate_bias(S, y, grid, passes=3):
    """Tìm bias cộng thêm cho điểm từng lớp (lớp 0 cố định = 0) để tối đa macro-F1 trên (S, y).
    Chỉ gọi với tập VALID"""
    def score(b):
        cm = conf_matrix(y, (S + b).argmax(1))
        tp = np.diag(cm).astype(np.float64)
        pr = tp / np.maximum(cm.sum(0), 1)
        rc = tp / np.maximum(cm.sum(1), 1)
        return float((2 * pr * rc / np.maximum(pr + rc, 1e-12)).mean())

    b = np.zeros(S.shape[1], dtype=np.float32)
    best = score(b)
    for _ in range(passes):
        improved = False
        for k in range(1, S.shape[1]):
            for delta in grid:
                bb = b.copy()
                bb[k] = delta
                f = score(bb)
                if f > best + 1e-6:
                    best, b, improved = f, bb, True
        if not improved:
            break
    return b, best


# HUẤN LUYỆN & BÁO CÁO
def main():
    random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = CFG["OUT_DIR"]
    os.makedirs(out, exist_ok=True)
    print(f"-> Thiết bị: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    # Dữ liệu
    print("\n1. NẠP & TIỀN XỬ LÝ DỮ LIỆU")
    data, feat_names = load_all_splits()

    def take(split):
        X, atk, folder = data.pop(split)
        return X, np.where(atk == 0, 0, folder.astype(np.int64) + 1)

    Xtr, ytr = take("train")
    Xva, yva = take("valid")
    Xte, yte = take("test")
    del data
    gc.collect()

    unit_name = {1.0: 's', 1e-3: 'ms', 1e-6: 'us', 1e-9: 'ns'}.get(STATE['time_scale'], str(STATE['time_scale']))
    print(f"\nĐơn vị thời gian đã dùng: {unit_name} | "
          f"id người nhận: {'có' if STATE['has_rid'] else 'không'}")
    print(f"{len(feat_names)} đặc trưng: {feat_names}")
    for name, yy in (("train", ytr), ("valid", yva), ("test", yte)):
        c = np.bincount(yy, minlength=K)
        print(f"{name:>5}: {len(yy):,} mẫu | " + ", ".join(f"{CLASS_NAMES[i]}={c[i]:,}" for i in range(K)))
        assert (c > 0).all(), f"Tập {name} thiếu lớp: {[CLASS_NAMES[i] for i in range(K) if c[i] == 0]}"

    if CFG["BENIGN_TRAIN_RATIO"]:
        b_idx = np.where(ytr == 0)[0]
        a_idx = np.where(ytr != 0)[0]
        cap = int(np.bincount(ytr[a_idx]).max() * CFG["BENIGN_TRAIN_RATIO"])
        if len(b_idx) > cap:
            kept = np.random.choice(b_idx, cap, replace=False)
            sel_idx = np.concatenate([a_idx, kept])
            np.random.shuffle(sel_idx)
            Xtr, ytr = Xtr[sel_idx], ytr[sel_idx]
            print(f"-> Giảm benign trong train còn {cap:,} mẫu (tập valid/test giữ nguyên phân phối thật)")

    log_idx = [feat_names.index(c) for c in CFG["LOG_COLS"] if c in feat_names]

    def signed_log(X):
        for j in log_idx:
            c = X[:, j]
            X[:, j] = np.sign(c) * np.log1p(np.abs(c))
        return X

    def mean_std(a, chunk=1_000_000):
        s = np.zeros(a.shape[1]); ss = np.zeros(a.shape[1])
        for i in range(0, len(a), chunk):
            c = a[i:i + chunk].astype(np.float64)
            s += c.sum(0); ss += (c ** 2).sum(0)
        m = s / len(a)
        sd = np.sqrt(np.maximum(ss / len(a) - m ** 2, 0.0))
        sd[sd < 1e-8] = 1.0
        return m, sd

    Xtr, Xva, Xte = signed_log(Xtr), signed_log(Xva), signed_log(Xte)
    mean, std = mean_std(Xtr)                                  # chỉ từ train
    mean32, std32, clip = mean.astype(np.float32), std.astype(np.float32), CFG["SCALE_CLIP"]

    def standardize(a):
        a -= mean32
        a /= std32
        np.clip(a, -clip, clip, out=a)
        return a

    Xtr, Xva, Xte = standardize(Xtr), standardize(Xva), standardize(Xte)

    sel = np.random.choice(len(ytr), min(200_000, len(ytr)), replace=False)
    dt = DecisionTreeClassifier(max_depth=3, random_state=CFG["SEED"]).fit(Xtr[sel], ytr[sel])
    top = np.argsort(dt.feature_importances_)[::-1][:5]
    dt_acc = dt.score(Xtr[sel], ytr[sel])
    print(f"\n[Leakage check] cây depth=3 đạt accuracy {dt_acc:.4f}; top feature: "
          f"{[(feat_names[i], round(float(dt.feature_importances_[i]), 3)) for i in top]}")
    if dt_acc > 0.98:
        print("!!! CẢNH BÁO: cây rất nông phân loại gần hoàn hảo -> kiểm tra top feature trước khi tin kết quả.")

    w_cls = class_weights_from_labels(ytr, K)                  # w_k = sqrt(n_max/n_k), trung bình = 1
    print("Trọng số reward:", dict(zip(CLASS_NAMES, w_cls.round(3).tolist())))

    # Huấn luyện
    print("\n2. HUẤN LUYỆN DOUBLE DQN")
    env = TorchBatchEnv(Xtr, ytr, w_cls, dev, CFG["ENV_BATCH"])
    agent = DoubleDQNAgent(Xtr.shape[1], K, dev)
    q_values = agent.q_values
    N = env.N
    steps_per_epoch = env.steps_per_epoch
    eps_decay_steps = max(1, int(CFG["EPS_DECAY_FRAC"] * steps_per_epoch * CFG["EPOCHS"]))
    print(f"N_train={N:,} | {steps_per_epoch:,} bước/epoch | tối đa {CFG['EPOCHS']} epoch | GAMMA={CFG['GAMMA']}")

    best_f1, bad_epochs, gstep, history = -1.0, 0, 0, []
    best_path = os.path.join(out, "best_ddqn.pt")

    for epoch in range(1, CFG["EPOCHS"] + 1):
        t0 = time.time()
        env.reset()
        loss_sum, n_upd = torch.zeros((), device=dev), 0

        for s_i in range(steps_per_epoch):
            gstep += 1
            eps = max(CFG["EPS_END"], CFG["EPS_START"] - (CFG["EPS_START"] - CFG["EPS_END"]) * gstep / eps_decay_steps)
            idx, nidx = env.batch(s_i)
            a = agent.act(env.X[idx], eps)
            r = env.reward(idx, a)
            agent.buffer.push(idx, a, r, nidx)

            if agent.buffer.size >= CFG["BATCH"]:
                for _ in range(CFG["UPDATES_PER_STEP"]):
                    loss_sum += agent.update(env.X)
                    n_upd += 1

            if gstep % CFG["TARGET_UPDATE"] == 0:
                agent.sync_target()

        agent.end_epoch()

        cm_v = conf_matrix(yva, q_values(Xva).argmax(1))
        mv = metrics_from_cm(cm_v)
        avg_loss = loss_sum.item() / max(n_upd, 1)
        history.append(dict(epoch=epoch, eps=eps, lr=agent.lr, loss=avg_loss,
                            val_acc=mv["accuracy"], val_macro_f1=mv["macro"]["f1"]))
        tag = ""
        if mv["macro"]["f1"] > best_f1:
            best_f1, bad_epochs = mv["macro"]["f1"], 0
            agent.save(best_path)
            tag = "  [lưu model tốt nhất]"
        else:
            bad_epochs += 1
        print(f"Epoch {epoch:2d}/{CFG['EPOCHS']} | eps {eps:.3f} | loss {avg_loss:.4f} | "
              f"val_acc {mv['accuracy']:.4f} | val_macroF1 {mv['macro']['f1']:.4f} | {time.time() - t0:.0f}s{tag}")
        if CFG["PATIENCE"] and bad_epochs >= CFG["PATIENCE"]:
            print(f"-> Dừng sớm: val macro-F1 không cải thiện sau {CFG['PATIENCE']} epoch.")
            break

    agent.load(best_path)

    # Đánh giá DDQN
    print("\n3. ĐÁNH GIÁ")
    Qva, Qte = q_values(Xva), q_values(Xte)
    res = {}
    res["DDQN (raw)"] = print_report("DDQN - VALID (raw)", conf_matrix(yva, Qva.argmax(1)))
    cm_te_raw = conf_matrix(yte, Qte.argmax(1))
    res["DDQN (raw) test"] = print_report("DDQN - TEST (raw)", cm_te_raw)
    final_cm, bias_q = cm_te_raw, np.zeros(K, dtype=np.float32)
    if CFG["CALIBRATE"]:
        bias_q, f_cal = calibrate_bias(Qva, yva, np.linspace(-1.5, 1.5, 31))
        print(f"\n[Hiệu chỉnh bias trên VALID] bias = {dict(zip(CLASS_NAMES, bias_q.round(2).tolist()))}, "
              f"val macro-F1 {best_f1:.4f} -> {f_cal:.4f}")
        final_cm = conf_matrix(yte, (Qte + bias_q).argmax(1))
        res["DDQN (calibrated) test"] = print_report("DDQN - TEST (đã hiệu chỉnh bias trên valid)", final_cm)

    sub = np.random.choice(len(ytr), min(200_000, len(ytr)), replace=False)
    train_acc = float((q_values(Xtr[sub]).argmax(1) == ytr[sub]).mean())
    print(f"\nTrain accuracy (mẫu con): {train_acc:.4f}   <- so với test accuracy để phát hiện overfit")

    # Baseline GBDT cùng đặc trưng
    if CFG["RUN_BASELINE"]:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("\n4. BASELINE GBDT (cùng đặc trưng)")
        bi = np.random.choice(len(ytr), min(CFG["BASELINE_ROWS"], len(ytr)), replace=False)
        t0 = time.time()
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=CFG["SEED"])
        clf.fit(Xtr[bi], ytr[bi])
        assert list(clf.classes_) == list(range(K))

        def log_proba(Xn, bs=500_000):
            return np.concatenate([np.log(clf.predict_proba(Xn[i:i + bs]) + 1e-9).astype(np.float32)
                                   for i in range(0, len(Xn), bs)])

        Lva, Lte = log_proba(Xva), log_proba(Xte)
        print(f"(huấn luyện + dự đoán: {time.time() - t0:.0f}s)")
        res["GBDT (raw) test"] = print_report("GBDT - TEST (raw)", conf_matrix(yte, Lte.argmax(1)))
        if CFG["CALIBRATE"]:
            bias_g, f_g = calibrate_bias(Lva, yva, np.linspace(-4, 4, 33))
            print(f"[Hiệu chỉnh bias GBDT trên VALID] val macro-F1 -> {f_g:.4f}")
            res["GBDT (calibrated) test"] = print_report("GBDT - TEST (đã hiệu chỉnh trên valid)",
                                                         conf_matrix(yte, (Lte + bias_g).argmax(1)))

    # Bảng tóm tắt kết quả
    rows = []
    for k_, v in res.items():
        if k_.endswith("test"):
            rows.append(dict(model=k_.replace(" test", ""), accuracy=v["accuracy"],
                             macro_precision=v["macro"]["precision"], macro_recall=v["macro"]["recall"],
                             macro_f1=v["macro"]["f1"], weighted_f1=v["weighted"]["f1"],
                             attack_vs_benign_f1=v["binary_attack_vs_benign"]["f1"]))
    summary = pd.DataFrame(rows)
    print("\nBẢNG TÓM TẮT TRÊN TẬP TEST")
    print(summary.round(4).to_string(index=False))

    summary.to_csv(os.path.join(out, "summary_test.csv"), index=False)
    per_class = pd.DataFrame({f"{m_}|{c}": v for m_, mv_ in res.items() if m_.endswith("test")
                              for c, v in mv_["per_class"].items()}).T
    per_class.to_csv(os.path.join(out, "per_class_test.csv"))
    pd.DataFrame(history).to_csv(os.path.join(out, "train_history.csv"), index=False)
    pd.DataFrame(final_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(os.path.join(out, "confusion_matrix_test.csv"))
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(dict(class_names=CLASS_NAMES, results=res, train_accuracy_subsample=train_acc, cfg={
            k_: (list(v) if isinstance(v, tuple) else v) for k_, v in CFG.items()}), fh, indent=2)
    with open(os.path.join(out, "preprocess_meta.json"), "w") as fh:
        json.dump(dict(feature_names=feat_names, scaler_mean=mean.tolist(), scaler_scale=std.tolist(), clip=clip,
                       log_cols=[feat_names[j] for j in log_idx], window=CFG["WINDOW"],
                       time_scale=STATE["time_scale"], has_rid=bool(STATE["has_rid"]), class_names=CLASS_NAMES,
                       hidden=list(CFG["HIDDEN"]), class_bias=bias_q.tolist()), fh, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        h = pd.DataFrame(history)
        ax[0].plot(h.epoch, h.val_acc, label="val accuracy")
        ax[0].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
        ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title("Validation")
        norm_cm = final_cm / np.maximum(final_cm.sum(1, keepdims=True), 1)
        ax[1].imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
        ax[1].set_xticks(range(K)); ax[1].set_yticks(range(K))
        ax[1].set_xticklabels(CLASS_NAMES, rotation=45, ha="right"); ax[1].set_yticklabels(CLASS_NAMES)
        for i in range(K):
            for j in range(K):
                ax[1].text(j, i, f"{norm_cm[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax[1].set_title("Confusion matrix chuẩn hoá theo hàng (test)")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "results.png"), dpi=130)
        plt.show()
    except Exception as e:
        print("Bỏ qua vẽ biểu đồ:", e)

    print(f"\n=> Hoàn tất")

if __name__ == "__main__":
    main()