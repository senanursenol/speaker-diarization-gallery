import argparse
from pathlib import Path
import json
import pandas as pd
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

def load_rttm(path: Path) -> Annotation:
    ann = Annotation(uri=path.stem)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if not p or p[0] != "SPEAKER":
                continue

            start = float(p[3]); dur = float(p[4]); spk = p[7]
            ann[Segment(start, start + dur)] = spk
    return ann

def load_uem(uem_path: Path):
    uem = {}
    with open(uem_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 4:
                continue
            uri, _chan, start, end = p[0], p[1], float(p[2]), float(p[3])
            tl = uem.get(uri, Timeline(uri=uri))
            tl.add(Segment(start, end))
            uem[uri] = tl
    return uem

def pct(x): 
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"

def main():
    ap = argparse.ArgumentParser(description="DER/JER (GT vs Model) toplu değerlendirme + JSON çıktısı")
    ap.add_argument("--gt-dir",        default="audios", help="GT RTTM klasörü (vars: audios)")
    ap.add_argument("--pred-dir",      default=".",      help="Model RTTM klasörü (vars: .)")
    ap.add_argument("--pred-suffix",   default=".pyannote.rttm", help="Model RTTM son eki (vars: .pyannote.rttm)")
    ap.add_argument("--out-dir",       default="data/metrics",   help="JSON çıktı klasörü (vars: data/metrics)")
    ap.add_argument("--collar",        type=float, default=0.0,  help="Collar toleransı, saniye (vars: 0.0)")
    ap.add_argument("--skip-overlap",  action="store_true", help="Örtüşen konuşmayı değerlendirme dışı bırak")
    ap.add_argument("--uem",           type=str, default=None,   help="UEM dosya yolu (opsiyonel)")
    args = ap.parse_args()

    gt_dir   = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    suffix   = args.pred_suffix
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    uem = load_uem(Path(args.uem)) if args.uem else None

    pred_files = sorted(pred_dir.glob(f"*{suffix}"))
    if not pred_files:
        print(f"Model RTTM bulunamadı: {pred_dir}/*{suffix}")
        return

    der_metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    jer_metric = JaccardErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)

    rows = []
    for pred in pred_files:
        stem = pred.name.replace(suffix, "")
        gt = gt_dir / f"{stem}.rttm"
        if not gt.exists():
            print(f"[Uyarı] GT yok, atlandı: {gt}")
            continue

        ref = load_rttm(gt)
        hyp = load_rttm(pred)

        this_uem = None
        if uem:
            this_uem = uem.get(ref.uri)

        if this_uem is not None:
            der_val = der_metric(ref, hyp, uem=this_uem)
            jer_val = jer_metric(ref, hyp, uem=this_uem)
        else:
            der_val = der_metric(ref, hyp)
            jer_val = jer_metric(ref, hyp)

        out_json = {
            "der": float(der_val),
            "jer": float(jer_val),
            "n_ref": len(set(ref.labels())),
            "n_sys": len(set(hyp.labels())),
            "window_sec": None if args.collar is None else float(args.collar),
            "notes": f"collar={args.collar}, skip_overlap={args.skip_overlap}, "
                     f"uem={'yes' if this_uem is not None else 'no'}"
        }
        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

        rows.append({"Audio": stem, "DER_num": der_val, "JER_num": jer_val, "Hyp": pred.name})

    if not rows:
        print("Skorlanacak eşleşme yok.")
        return

    df = pd.DataFrame(rows).sort_values("DER_num").reset_index(drop=True)
    df_view = df.copy()
    df_view["DER"] = df_view["DER_num"].apply(pct)
    df_view["JER"] = df_view["JER_num"].apply(pct)
    df_view = df_view[["Audio", "DER", "JER", "Hyp"]]

    mean_der = df["DER_num"].mean()
    mean_jer = df["JER_num"].mean()

    print("\n--- Çoklu Dosya DER/JER (yüzde) ---")
    print(df_view.to_string(index=False))
    print(f"\nOrtalama  DER: {pct(mean_der)}   JER: {pct(mean_jer)}")

    out_csv = pred_dir / "der_jer_batch_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Kaydedildi: {out_csv}")

if __name__ == "__main__":
    main()
