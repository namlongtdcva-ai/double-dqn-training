"""
- Tiền xử lý & trích xuất đặc trưng
- Logic: giữ nguyên từng dòng; CFG/ATTACKS/CLASS_NAMES/K/SPLITS/STATE nằm ngay đầu file này (các file khác sẽ import từ đây).
- Quy trình: đọc JSON theo luồng -> gom theo thông điệp duy nhất (sender, sendTime) -> 24 đặc trưng
(phía người gửi tính trên bảng thông điệp rồi ghép lại từng dòng; phía người nhận tính theo dòng).
"""
import os
import gc
import re
import json
import math
import time
from array import array
from collections import Counter

import numpy as np
import pandas as pd


#CẤU HÌNH
ON_KAGGLE = os.path.isdir("/kaggle/working")
_BASE_OUT = "/kaggle/working" if ON_KAGGLE else "outputs"

CFG = dict(
    DATASET_NAME="ddqndataset12",
    DATA_DIR=None,                       # None = ./data nếu chạy local, ngược lại tự tìm trong /kaggle/input
    OUT_DIR=_BASE_OUT,
    CACHE_DIR=os.path.join(_BASE_OUT, "cache"),
    USE_CACHE=True,
    # Đặc trưng
    WINDOW=5,                            # cửa sổ rolling theo từng xe gửi (trên bảng thông điệp)
    TIME_UNIT="auto",                    # đơn vị của rcvTime/sendTime
    MIN_DT=0.01,                         # giây; chặn mẫu số khi tính vận tốc/gia tốc
    LOG_COLS=["send_rate", "dpos", "dx", "dy", "spd_calc", "spd_err", "dspd", "acl_calc", "acl_err",
              "spd_err_rmean", "spd_err_rstd", "acl_err_rmean", "acl_err_rstd", "dpos_rmean", "dpos_rstd",
              "latency", "dist_to_rx", "rcv_gap"],
    SCALE_CLIP=20.0,
    SEED=42,
    BENIGN_TRAIN_RATIO=None,            
    # Double DQN
    HIDDEN=(256, 128, 64),
    GAMMA=0.1,
    LR=3e-4,
    COSINE_LR=True,
    EPOCHS=15,
    PATIENCE=4,                      
    ENV_BATCH=2048,
    BATCH=2048,
    UPDATES_PER_STEP=2,
    BUFFER_SIZE=500_000,
    TARGET_UPDATE=500,
    EPS_START=1.0, EPS_END=0.05, EPS_DECAY_FRAC=0.5,
    # Đánh giá
    CALIBRATE=True,                      # hiệu chỉnh bias từng lớp trên valid để tối đa macro-F1
    RUN_BASELINE=True,                   # baseline GBDT cùng đặc trưng để đối chiếu trong báo cáo
    BASELINE_ROWS=1_000_000,
)
ATTACKS = ["dos", "feigned braking", "random position", "random speed", "traffic congestion"]
CLASS_NAMES = ["benign"] + ATTACKS
K = len(CLASS_NAMES)
SPLITS = ("train", "valid", "test")
NAN = float("nan")
STATE = {"time_scale": None, "has_rid": None}


# PHẦN 1: ĐỌC DỮ LIỆU
def norm(s):
    return s.lower().replace("_", " ").replace("-", " ")


def find_dataset_dir(name):
    for c in (f"/kaggle/input/datasets/longnamle/{name}", f"/kaggle/input/{name}"):
        if os.path.isdir(c):
            return c
    base = "/kaggle/input"
    for root, dirs, _ in os.walk(base):
        if root.count(os.sep) - base.count(os.sep) >= 4:
            dirs[:] = []
            continue
        if name in dirs:
            return os.path.join(root, name)
    top = os.listdir(base) if os.path.isdir(base) else []
    raise FileNotFoundError(f"Không thấy dataset '{name}'. Nội dung /kaggle/input: {top}")


def resolve_data_dir():
    """CFG['DATA_DIR'] nếu có; local: ./data; Kaggle: tự tìm theo DATASET_NAME."""
    if CFG["DATA_DIR"]:
        return CFG["DATA_DIR"]
    if not ON_KAGGLE and os.path.isdir("data"):
        return "data"
    return find_dataset_dir(CFG["DATASET_NAME"])


def iter_records(path, chunk_chars=1 << 24, max_record_chars=1 << 26, stats=None):
    """Đọc luồng từng bản ghi JSON"""
    if stats is not None:
        stats["truncated_chars"] = 0
    dec = json.JSONDecoder()
    skip = " \t\r\n,[]"
    buf, eof = "", False
    with open(path, "r", encoding="utf-8-sig") as f:
        while not eof:
            chunk = f.read(chunk_chars)
            eof = not chunk
            buf += chunk
            pos, n = 0, len(buf)
            while True:
                while pos < n and buf[pos] in skip:
                    pos += 1
                if pos >= n:
                    break
                try:
                    obj, end = dec.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if not eof:
                        if n - pos > max_record_chars:
                            raise ValueError(f"{path}: bản ghi hỏng hoặc quá lớn tại ký tự {pos}")
                        break
                    if stats is not None:
                        stats["truncated_chars"] = n - pos
                    print(f"[cảnh báo] {os.path.basename(path)}: phần cuối file không phải JSON hợp lệ, "
                          f"bỏ qua {n - pos:,} ký tự")
                    return
                yield obj
                pos = end
            buf = buf[pos:]


def _num(v):
    t = type(v)
    if t is float or t is int or t is bool:
        return float(v)
    if t in (list, tuple):
        try:
            return math.sqrt(sum(float(a) ** 2 for a in v))
        except (TypeError, ValueError):
            return NAN
    if t is dict:
        try:
            return math.sqrt(sum(float(a) ** 2 for a in v.values()))
        except (TypeError, ValueError):
            return NAN
    if t is str:
        try:
            return float(v)
        except ValueError:
            return NAN
    return NAN


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _xy(v):
    """pos có thể là list [x,y,z], dict {x,y,z} hoặc chuỗi 'x,y,z' / '[x, y, z]'."""
    if isinstance(v, (list, tuple)):
        vals = v
    elif isinstance(v, dict):
        vals = [v.get("x"), v.get("y")] if "x" in v else list(v.values())
    elif isinstance(v, str):
        vals = _FLOAT_RE.findall(v)
    else:
        return NAN, NAN
    out = []
    for i in range(2):
        try:
            out.append(float(vals[i]))
        except (IndexError, TypeError, ValueError):
            out.append(NAN)
    return out[0], out[1]


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def extract_file(path):
    """Đọc 1 file theo luồng -> dict các mảng numpy"""
    keys = ("rcv", "snd", "spd", "acl", "sx", "sy", "rx", "ry")
    cols = {k: array("d") for k in keys}
    sids, rids, atk = array("i"), array("i"), array("b")
    sid_map, rid_map, raw_atk = {}, {}, Counter()
    n = miss_sid = miss_rid = miss_atk = 0

    for item in iter_records(path):
        if not isinstance(item, dict):
            continue
        n += 1
        s = item.get("sender")
        s = s if isinstance(s, dict) else {}
        r = item.get("receiver")
        r = r if isinstance(r, dict) else {}
        cols["rcv"].append(_num(item.get("rcvTime")))
        cols["snd"].append(_num(item.get("sendTime")))
        cols["spd"].append(_num(s.get("spd")))
        cols["acl"].append(_num(s.get("acl")))
        x, y = _xy(s.get("pos"))
        cols["sx"].append(x); cols["sy"].append(y)
        x, y = _xy(r.get("pos"))
        cols["rx"].append(x); cols["ry"].append(y)

        sid = _first(item.get("sender_id"), item.get("sender_alias"), s.get("id"), s.get("alias"))
        if sid is None:
            miss_sid += 1
            sid = -1
        if not isinstance(sid, (int, str)):
            sid = str(sid)
        sids.append(sid_map.setdefault(sid, len(sid_map)))

        rid = _first(item.get("receiver_id"), item.get("receiver_alias"), r.get("id"), r.get("alias"))
        if rid is None:
            miss_rid += 1
            rid = -1
        if not isinstance(rid, (int, str)):
            rid = str(rid)
        rids.append(rid_map.setdefault(rid, len(rid_map)))

        a = _first(item.get("attacker"), s.get("attacker"))
        if a is None:
            miss_atk += 1
            a = 0
        key = str(a)
        if key in raw_atk or len(raw_atk) < 50:
            raw_atk[key] += 1
        v = _num(a)
        atk.append(1 if (v == v and v != 0.0) else 0)

    if n == 0:
        raise RuntimeError(f"{path}: không đọc được bản ghi nào")
    if miss_sid > 0.5 * n:
        raise RuntimeError(f"{path}: >50% bản ghi thiếu id người gửi (sender_id/sender_alias). "
                           f"Đặc trưng theo từng xe sẽ sai, kiểm tra tên trường.")
    if miss_atk == n:
        raise RuntimeError(f"{path}: không có trường 'attacker' ở bản ghi nào, nhãn sẽ sai.")

    has_rid = miss_rid <= 0.5 * n
    if STATE["has_rid"] is None:
        STATE["has_rid"] = has_rid
    elif STATE["has_rid"] != has_rid:
        raise RuntimeError(f"{path}: khác các file trước về việc có id người nhận (has_rid={has_rid}).")

    raw = {k: np.frombuffer(v, dtype=np.float64) for k, v in cols.items()}
    raw["sid"] = np.frombuffer(sids, dtype=np.intc)
    raw["rid"] = np.frombuffer(rids, dtype=np.intc)
    raw["atk"] = np.frombuffer(atk, dtype=np.int8)
    print(f"   {n:,} bản ghi | {len(sid_map):,} xe gửi | {len(rid_map) if has_rid else 0:,} xe nhận | "
          f"attacker={dict(raw_atk.most_common(6))} | thiếu id gửi {miss_sid:,}, thiếu nhãn {miss_atk:,} | "
          f"NaN: sendTime {np.isnan(raw['snd']).mean():.2%}, spd {np.isnan(raw['spd']).mean():.2%}, "
          f"pos {np.isnan(raw['sx']).mean():.2%}")
    return raw


# PHẦN 2: ĐẶC TRƯNG
def decide_time_scale(raw):
    fixed = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    u = CFG["TIME_UNIT"]
    if u in fixed:
        return fixed[u]
    o = np.lexsort((raw["snd"], raw["sid"]))
    sid, snd = raw["sid"][o], raw["snd"][o]
    d = np.diff(snd)
    d = d[(sid[1:] == sid[:-1]) & (d > 0)]
    if len(d) == 0:
        raise RuntimeError("Không ước lượng được đơn vị thời gian, hãy đặt CFG['TIME_UNIT'] = 's'/'ms'/'us'/'ns'.")
    med, p90 = float(np.median(d)), float(np.percentile(d, 90))
    # p90 (không phải median) để DoS gửi dày không làm lệch ước lượng; beacon thường ~ 0.1-1 giây
    if p90 >= 1e8:
        name, scale = "nano-giây", 1e-9
    elif p90 >= 1e5:
        name, scale = "micro-giây", 1e-6
    elif p90 > 20:
        name, scale = "mili-giây", 1e-3
    else:
        name, scale = "giây", 1.0
    print(f"   [đơn vị thời gian] khoảng cách giữa 2 thông điệp liên tiếp của 1 xe: median {med:g}, p90 {p90:g} "
          f"-> coi là {name} (nếu sai, đặt CFG['TIME_UNIT'] = 's'/'ms'/'us'/'ns')")
    return scale


def build_features(raw, window):
    """Đặc trưng phía người gửi tính trên bảng thông điệp (loại trùng sender+sendTime), ghép lại cho từng dòng;
    đặc trưng phía người nhận (latency, khoảng cách tới người nhận, khoảng cách giữa 2 lần nhận) tính theo dòng."""
    if STATE["time_scale"] is None:
        STATE["time_scale"] = decide_time_scale(raw)
    ts = STATE["time_scale"]
    n = len(raw["sid"])
    if np.isnan(raw["snd"]).mean() > 0.01:
        raise RuntimeError("Hơn 1% bản ghi thiếu sendTime, không thể gom thông điệp.")

    order = np.lexsort((raw["snd"], raw["sid"]))          # theo xe gửi, rồi thời điểm gửi
    g = {k: v[order] for k, v in raw.items()}
    sid, snd, rcv = g["sid"], g["snd"] * ts, g["rcv"] * ts

    chg = np.ones(n, dtype=bool)
    chg[1:] = (sid[1:] != sid[:-1]) | (snd[1:] != snd[:-1])
    msg_idx = np.cumsum(chg) - 1
    first = np.flatnonzero(chg)
    n_msg = len(first)
    m_sid, m_snd = sid[first], snd[first]
    m_sx, m_sy, m_spd, m_acl = g["sx"][first], g["sy"][first], g["spd"][first], g["acl"][first]
    same = m_sid[1:] == m_sid[:-1]
    print(f"   {n:,} dòng -> {n_msg:,} thông điệp duy nhất (trung bình {n / n_msg:.2f} dòng/thông điệp)")

    def gdiff(a):
        d = np.full(len(a), np.nan)
        d[1:] = a[1:] - a[:-1]
        d[1:][~same] = np.nan
        return d

    send_int = gdiff(m_snd)
    dx, dy = gdiff(m_sx), gdiff(m_sy)
    dpos = np.hypot(dx, dy)
    dt = np.maximum(send_int, CFG["MIN_DT"])
    spd_calc = dpos / dt
    spd_err = np.abs(spd_calc - m_spd)
    dspd = gdiff(m_spd)
    acl_calc = dspd / dt
    acl_err = np.abs(acl_calc - m_acl)

    # kiểm tra vật lý: với thông điệp benign, vận tốc tính từ vị trí phải xấp xỉ vận tốc khai báo nếu đơn vị đúng
    ok = (g["atk"][first] == 0) & (m_spd > 1.0) & (send_int > 0) & np.isfinite(dpos)
    if ok.sum() > 100:
        ratio = float(np.median(dpos[ok] / send_int[ok] / m_spd[ok]))
        print(f"   [kiểm tra vật lý] median(vận tốc tính từ vị trí / vận tốc khai báo) của thông điệp benign = "
              f"{ratio:.3f} (xấp xỉ 1 nếu đơn vị thời gian và vận tốc/vị trí nhất quán)")
        if not 0.5 < ratio < 2.0:
            print("   !!! CẢNH BÁO: tỉ số lệch xa 1 -> kiểm tra CFG['TIME_UNIT'] (hoặc đơn vị vận tốc/vị trí, vd km/h).")

    mt = pd.DataFrame({"sid": m_sid, "spd": m_spd, "spd_err": spd_err, "send_interval": send_int,
                       "acl_err": acl_err, "dpos": dpos})
    grp = mt.groupby("sid", sort=False)
    roll = {}
    for c in ("spd", "spd_err", "send_interval", "acl_err", "dpos"):
        rr = grp[c].rolling(window, min_periods=1)
        roll[f"{c}_rmean"] = rr.mean().reset_index(level=0, drop=True).reindex(mt.index).to_numpy()
        roll[f"{c}_rstd"] = rr.std().reset_index(level=0, drop=True).reindex(mt.index).to_numpy()
    del mt, grp

    msg_feats = {"sender.spd": m_spd, "sender.acl": m_acl, "send_interval": send_int, "send_rate": 1.0 / dt,
                 "dpos": dpos, "dx": dx, "dy": dy, "spd_calc": spd_calc, "spd_err": spd_err,
                 "dspd": dspd, "acl_calc": acl_calc, "acl_err": acl_err, **roll}
    names = list(msg_feats)
    M = np.column_stack([msg_feats[k] for k in names]).astype(np.float32)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    parts = [M[msg_idx]]
    del M

    row_cols = [(rcv - snd), np.hypot(g["sx"] - g["rx"], g["sy"] - g["ry"])]
    names += ["latency", "dist_to_rx"]
    if STATE["has_rid"]:
        o2 = np.lexsort((rcv, sid, g["rid"]))
        r2, s2, t2 = g["rid"][o2], sid[o2], rcv[o2]
        same2 = (r2[1:] == r2[:-1]) & (s2[1:] == s2[:-1])
        gap_sorted = np.full(n, np.nan)
        gap_sorted[1:] = t2[1:] - t2[:-1]
        gap_sorted[1:][~same2] = np.nan
        gap = np.empty(n)
        gap[o2] = gap_sorted
        row_cols.append(gap)
        names.append("rcv_gap")
    R = np.column_stack(row_cols).astype(np.float32)
    parts.append(np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0))
    X = np.hstack(parts)
    return X, g["atk"].copy(), names


def discover_files(data_dir):
    found = {}
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        for f in sorted(files):
            if not f.lower().endswith(".json"):
                continue
            path = os.path.join(root, f)
            rel = norm(os.path.relpath(path, data_dir))
            hits = [a for a in ATTACKS if a in rel]
            if len(hits) != 1:
                print(f"[bỏ qua] không xác định được loại tấn công ({hits}): {rel}")
                continue
            name = norm(f)
            split = "train" if "train" in name else "valid" if "valid" in name else "test" if "test" in name else None
            if split is None:
                print(f"[bỏ qua] không xác định được train/valid/test từ tên file: {rel}")
                continue
            key = (ATTACKS.index(hits[0]), split)
            if key in found:
                raise ValueError(f"Trùng file cho {key}: {found[key]} và {path}")
            found[key] = path
    missing = [(ATTACKS[a], s) for a in range(len(ATTACKS)) for s in SPLITS if (a, s) not in found]
    if missing:
        raise RuntimeError(f"Thiếu file cho: {missing}. File tìm thấy: {sorted(found.values())}")
    return found


def load_all_splits():
    """Trả về (data, feature_names); data[split] = (X float32, atk int8, folder int8)."""
    tag = f"{CFG['DATASET_NAME']}_v4_w{CFG['WINDOW']}_{CFG['TIME_UNIT']}"
    cdir = CFG["CACHE_DIR"]
    need = [os.path.join(cdir, f"{tag}_{s}_{k}.npy") for s in SPLITS for k in ("X", "atk", "folder")]
    meta_path = os.path.join(cdir, f"{tag}_meta.json")
    if CFG["USE_CACHE"] and all(os.path.exists(p) for p in need + [meta_path]):
        print(f"-> Dùng cache đặc trưng trong {cdir}")
        data = {s: tuple(np.load(os.path.join(cdir, f"{tag}_{s}_{k}.npy")) for k in ("X", "atk", "folder"))
                for s in SPLITS}
        with open(meta_path) as fh:
            meta = json.load(fh)
        STATE.update(time_scale=meta["time_scale"], has_rid=meta["has_rid"])
        return data, meta["names"]

    data_dir = resolve_data_dir()
    print(f"Thư mục dữ liệu: {data_dir}")
    files = discover_files(data_dir)
    data, names = {}, None
    for split in SPLITS:
        Xs, As, Fs = [], [], []
        for a_id, a_name in enumerate(ATTACKS):
            path = files[(a_id, split)]
            t0 = time.time()
            print(f"-> [{split}] {a_name}: {os.path.relpath(path, data_dir)}")
            raw = extract_file(path)
            X, atk, nm = build_features(raw, CFG["WINDOW"])
            if names is not None and nm != names:
                raise RuntimeError("Danh sách đặc trưng khác nhau giữa các file.")
            names = nm
            del raw
            gc.collect()
            print(f"   -> {X.shape[0]:,} dòng x {X.shape[1]} đặc trưng | tỉ lệ attacker={atk.mean():.3f} | "
                  f"{time.time() - t0:.0f}s")
            Xs.append(X); As.append(atk); Fs.append(np.full(len(X), a_id, dtype=np.int8))
        data[split] = (np.concatenate(Xs), np.concatenate(As), np.concatenate(Fs))
        del Xs, As, Fs
        gc.collect()
    if CFG["USE_CACHE"]:
        os.makedirs(cdir, exist_ok=True)
        for s in SPLITS:
            for k, arr in zip(("X", "atk", "folder"), data[s]):
                np.save(os.path.join(cdir, f"{tag}_{s}_{k}.npy"), arr)
        with open(meta_path, "w") as fh:
            json.dump(dict(names=names, time_scale=STATE["time_scale"], has_rid=bool(STATE["has_rid"])), fh)
    return data, names


def preprocess_v2x_json(filepath, window_size=None):
    raw = extract_file(filepath)
    X, atk, names = build_features(raw, window_size or CFG["WINDOW"])
    df = pd.DataFrame(X, columns=names)
    df["attacker"] = atk.astype(int)
    return df

