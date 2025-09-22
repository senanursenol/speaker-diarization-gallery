import argparse
from pathlib import Path
import csv
import json

def main():
    ap = argparse.ArgumentParser(description="DER/JER CSV -> per-record JSON dönüştürücü")
    ap.add_argument("--csv", required=True, help="evaluate.py çıktısı (der_jer_batch_results.csv)")
    ap.add_argument("--out-dir", default="data/metrics", help="JSON çıktı klasörü (vars: data/metrics)")
    ap.add_argument("--run", default=None, help="İsteğe bağlı: alt klasör etiketi (örn: run1)")
    ap.add_argument("--collar", type=float, default=0.0, help="Notlar için collar bilgisi")
    ap.add_argument("--skip-overlap", action="store_true", help="Notlar için skip-overlap bilgisi")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_base = Path(args.out_dir)
    out_dir = (out_base / args.run) if args.run else out_base
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("CSV boş görünüyor.")
        return

    for r in rows:
        rec_id = r["Audio"].strip()
        der = float(r["DER_num"])
        jer = float(r["JER_num"])
        hyp = r.get("Hyp", "")

        data = {
            "der": der,
            "jer": jer,
            "n_ref": None,           
            "n_sys": None,          
            "window_sec": args.collar,
            "notes": f"from CSV ({csv_path.name}); hyp={hyp}; collar={args.collar}; skip_overlap={args.skip_overlap}"
        }

        out_fp = out_dir / f"{rec_id}.json"
        with open(out_fp, "w", encoding="utf-8") as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)
        print(f"✔ {out_fp}")

    print(f"\nToplam {len(rows)} JSON yazıldı → {out_dir}")

if __name__ == "__main__":
    main()
