"""
- Chuẩn hoá file JSON log thành MỘT mảng JSON hợp lệ.
- Dùng bộ đọc iter_records() của preprocessing.py
  + Đọc được mảng nối dính "] [", JSON-lines, thiếu '[' hoặc ']';
  + Nếu đuôi file bị cụt giữa chừng, bản ghi dở dang bị bỏ và được báo cáo;
  + Ghi theo luồng ra file tạm rồi mới thay thế (để không làm hỏng file gốc giữa chừng).
"""
import os
import sys
import json

from preprocessing import iter_records


def normalize_json_file(path, out_path=None, inplace=False, backup=False):
    """Ghi lại `path` thành mảng JSON hợp lệ. Trả về (số_bản_ghi, số_ký_tự_đuôi_bị_bỏ, đường_dẫn_đầu_ra)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    final = path if inplace else (out_path or os.path.splitext(path)[0] + ".fixed.json")
    tmp = final + ".tmp"
    stats, n = {}, 0
    try:
        with open(tmp, "w", encoding="utf-8") as fout:
            fout.write("[")
            for rec in iter_records(path, stats=stats):
                if n:
                    fout.write(",\n")
                fout.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
                n += 1
            fout.write("]")
        if inplace and backup:
            os.replace(path, path + ".bak")
        os.replace(tmp, final)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return n, stats.get("truncated_chars", 0), final


def count_records(path):
    """Đếm bản ghi đọc được và số ký tự đuôi bị bỏ"""
    stats, n = {}, 0
    for _ in iter_records(path, stats=stats):
        n += 1
    return n, stats.get("truncated_chars", 0)


def normalize_dir(data_dir, inplace=False, backup=False):
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(".json") and not f.lower().endswith((".fixed.json", ".tmp")):
                p = os.path.join(root, f)
                n, cut, dst = normalize_json_file(p, inplace=inplace, backup=backup)
                note = f" | bỏ {cut:,} ký tự đuôi dở dang" if cut else ""
                print(f"=> {p}: {n:,} bản ghi -> {dst}{note}")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    normalize_dir(args[0] if args else "data", inplace="--inplace" in sys.argv, backup="--backup" in sys.argv)