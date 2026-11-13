"""
- Kiểm tra / sửa file JSON bị cụt đuôi.
- Dùng bộ đọc của pipeline (iter_records): chỉ giữ các bản ghi đọc trọn vẹn, bỏ phần đuôi dở dang,
ghi ra file tạm rồi mới thay thế.
- Cách dùng:
  python fix.py <file.json> [file2.json ...]            # chỉ kiểm tra: đếm bản ghi, báo đuôi dở dang
  python fix.py <file.json> --repair                    # ghi "<tên>.fixed.json"
  python fix.py <file.json> --repair --inplace --backup # thay thế file gốc, giữ bản .bak
"""
import sys

from merge import count_records, normalize_json_file


def check_or_fix(path, repair=False, inplace=False, backup=False):
    if not repair:
        n, cut = count_records(path)
        print(f"{path}: {n:,} bản ghi đọc được | đuôi dở dang: {cut:,} ký tự" + ("  -> CẦN SỬA" if cut else "  -> OK"))
        return n, cut
    n, cut, dst = normalize_json_file(path, inplace=inplace, backup=backup)
    print(f"{path}: giữ {n:,} bản ghi, bỏ {cut:,} ký tự đuôi dở dang -> {dst}")
    return n, cut

if __name__ == "__main__":
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not files:
        sys.exit(__doc__)
    for f in files:
        check_or_fix(f, repair="--repair" in sys.argv, inplace="--inplace" in sys.argv, backup="--backup" in sys.argv)