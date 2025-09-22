from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT / "streamlit"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from helpers import write_rttm, make_wave_preview, update_manifest, human_time 

USAGE = "Kullanım: python scripts/run_pyannote.py <wav_path> [-o OUT_DIR] [--model NAME] [--hf-token TOKEN] [--min-spk N] [--max-spk N] [PP...]"

def get_pipeline(model_name: str, hf_token: Optional[str]):
    from pyannote.audio import Pipeline
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HuggingFace token gerekli. (arg --hf-token veya env HF_TOKEN)")
    return Pipeline.from_pretrained(model_name, use_auth_token=token)

def _ann_to_seglist(ann) -> List[Tuple[float,float,str]]:
    segs: List[Tuple[float,float,str]] = []
    for seg, _, lab in ann.itertracks(yield_label=True):
        segs.append((float(seg.start), float(seg.end), str(lab)))
    segs.sort(key=lambda x: (x[0], x[1]))
    return segs

def _merge_same_speaker_with_gap(segs, max_gap: float) -> List[Tuple[float,float,str]]:
    if not segs: return []
    out = [list(segs[0])]
    for s, e, l in segs[1:]:
        ps, pe, pl = out[-1]
        if l == pl and s <= pe + max_gap:
            out[-1][1] = max(pe, e)  
        else:
            out.append([s, e, l])
    return [(s, e, l) for s, e, l in out]

def _filter_tiny_segments(segs, min_seg: float, max_gap: float) -> List[Tuple[float,float,str]]:
    if not segs: return []
    out: List[List] = []
    for s, e, l in segs:
        dur = e - s
        if dur >= min_seg or not out:
            out.append([s, e, l]); continue
        ps, pe, pl = out[-1]
        if pl == l and (s - pe) <= max_gap:
            out[-1][1] = max(pe, e)
        else:
            out.append([s, e, l])
    res: List[List] = []
    for i, (s, e, l) in enumerate(out):
        dur = e - s
        if dur >= min_seg or not res:
            res.append([s, e, l])
        else:
            ps, pe, pl = res[-1]
            if pl == l and (s - pe) <= max_gap:
                res[-1][1] = max(pe, e)
            elif i+1 < len(out) and out[i+1][2] == l and (out[i+1][0] - e) <= max_gap:
                ns, ne, nl = out[i+1]
                out[i+1] = [s, max(e, ne), l]  
            else:
                res.append([s, e, l])
    return [(s, e, l) for s, e, l in res]

def _drop_short_speakers(segs, min_spk_total: float) -> List[Tuple[float,float,str]]:
    if min_spk_total <= 0: return segs
    totals: Dict[str, float] = {}
    for s, e, l in segs:
        totals[l] = totals.get(l, 0.0) + (e - s)
    keep = {l for l, t in totals.items() if t >= min_spk_total}
    return [(s, e, l) for (s, e, l) in segs if l in keep]

def _seglist_to_annotation(segs, uri: str):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=uri)
    for s, e, l in segs:
        ann[Segment(float(s), float(e))] = l
    return ann

def post_process_annotation(ann, uri: str, min_seg: float, max_gap: float, min_spk_total: float):
    segs = _ann_to_seglist(ann)
    segs = _merge_same_speaker_with_gap(segs, max_gap=max_gap)
    segs = _filter_tiny_segments(segs, min_seg=min_seg, max_gap=max_gap)
    segs = _merge_same_speaker_with_gap(segs, max_gap=max_gap)  
    segs = _drop_short_speakers(segs, min_spk_total=min_spk_total)
    return _seglist_to_annotation(segs, uri=uri)

def run_diarization(
    wav_path: Path,
    out_dir: Path,
    model_name: str,
    hf_token: Optional[str],
    params: Optional[Dict[str, Any]] = None,
    pp: Optional[Dict[str, float]] = None,
) -> Path:
    params = params or {}
    pipe = get_pipeline(model_name, hf_token)

    diar = pipe(
        wav_path,
        **{k: v for k, v in params.items() if v is not None}
    )

    if pp and not pp.get("disabled", False):
        diar = post_process_annotation(
            diar, uri=wav_path.stem,
            min_seg=float(pp.get("min_seg", 0.6)),
            max_gap=float(pp.get("max_gap", 0.2)),
            min_spk_total=float(pp.get("min_spk_total", 2.0)),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    rttm_out = out_dir / f"{wav_path.stem}.rttm"
    write_rttm(diar, uri=wav_path.stem, out_path=rttm_out)
    return rttm_out

def main():
    ap = argparse.ArgumentParser(description="Pyannote diarization runner + optional PP", usage=USAGE)
    ap.add_argument("wav_path", type=str, help="Girdi WAV dosyası")
    ap.add_argument("-o", "--out-dir", type=str, default="results", help="Çıktı klasörü (RTTM, önizleme vb.)")
    ap.add_argument("--model", type=str, default="pyannote/speaker-diarization-3.1", help="HF model adı")
    ap.add_argument("--hf-token", type=str, default=None, help="HF token (boşsa env HF_TOKEN)")
    ap.add_argument("--min-spk", type=int, default=None, help="min_speakers (örn: 3)")
    ap.add_argument("--max-spk", type=int, default=None, help="max_speakers (örn: 8)")

    # --- PP parametreleri
    ap.add_argument("--pp-min-seg", type=float, default=0.60, help="PP: min segment süresi (s)")
    ap.add_argument("--pp-max-gap", type=float, default=0.20, help="PP: aynı konuşmacı boşluk birleştirme eşiği (s)")
    ap.add_argument("--pp-min-spk-total", type=float, default=2.00, help="PP: konuşmacı toplam süre eşiği (s)")
    ap.add_argument("--no-pp", action="store_true", help="PP kapalı")

    ap.add_argument("--no-preview", action="store_true", help="Wave preview JSON üretme")
    args = ap.parse_args()

    wav_path = Path(args.wav_path).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV bulunamadı: {wav_path}")

    params = {"min_speakers": args.min_spk, "max_speakers": args.max_spk}
    pp_cfg = {
        "min_seg": args.pp_min_seg,
        "max_gap": args.pp_max_gap,
        "min_spk_total": args.pp_min_spk_total,
        "disabled": bool(args.no_pp),
    }

    print(f"[INFO] Model   : {args.model}")
    print(f"[INFO] Audio   : {wav_path.name}")
    print(f"[INFO] Out dir : {out_root}")
    print(f"[INFO] Spk rng : ({args.min_spk}, {args.max_spk}) | PP: "
          f"min_seg={args.pp_min_seg}s, max_gap={args.pp_max_gap}s, min_spk_total={args.pp_min_spk_total}s | disabled={args.no_pp}")

    rttm_path = run_diarization(
        wav_path=wav_path,
        out_dir=out_root,
        model_name=args.model,
        hf_token=args.hf_token,
        params=params,
        pp=pp_cfg,
    )
    print(f"[OK] RTTM kaydedildi -> {rttm_path}")

    duration = None
    if not args.no_preview:
        try:
            duration = make_wave_preview(wav_path, out_root)
            print(f"[OK] Preview JSON üretildi (süre: {human_time(duration)})")
        except Exception as e:
            print(f"[WARN] Preview üretilemedi: {e}")

    try:
        item = {
            "id": wav_path.stem,
            "wav": str(wav_path),
            "rttm": str(rttm_path),
            "duration_sec": float(duration) if duration is not None else None,
            "model": args.model,
            "params": {k: v for k, v in params.items() if v is not None},
            "pp": None if args.no_pp else {
                "min_seg": args.pp_min_seg,
                "max_gap": args.pp_max_gap,
                "min_spk_total": args.pp_min_spk_total
            }
        }
        update_manifest(out_root, item)
        print("[OK] Manifest güncellendi.")
    except Exception as e:
        print(f"[WARN] Manifest güncellenemedi: {e}")

if __name__ == "__main__":
    main()
