from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

from streamlit.helpers import write_rttm, make_wave_preview, update_manifest, human_time

USAGE = "Kullanım: python scripts/run_pyannote.py <wav_path> [-o OUT_DIR] [--model NAME] [--hf-token TOKEN] [--min-spk N] [--max-spk N]"

def get_pipeline(model_name: str, hf_token: Optional[str]):
    from pyannote.audio import Pipeline
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HuggingFace token gerekli. (arg --hf-token veya env HF_TOKEN)")
    return Pipeline.from_pretrained(model_name, use_auth_token=token)

def run_diarization(
    wav_path: Path,
    out_dir: Path,
    model_name: str,
    hf_token: Optional[str],
    params: Optional[Dict[str, Any]] = None,
) -> Path:
    params = params or {}
    pipe = get_pipeline(model_name, hf_token)
    diar = pipe(
        wav_path,
        **{k: v for k, v in params.items() if v is not None}
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rttm_out = out_dir / f"{wav_path.stem}.rttm"
    write_rttm(diar, uri=wav_path.stem, out_path=rttm_out)
    return rttm_out

def main():
    ap = argparse.ArgumentParser(description="Pyannote diarization runner", usage=USAGE)
    ap.add_argument("wav_path", type=str, help="Girdi WAV dosyası")
    ap.add_argument("-o", "--out-dir", type=str, default="results", help="Çıktı klasörü (RTTM, önizleme vb.)")
    ap.add_argument("--model", type=str, default="pyannote/speaker-diarization-3.1", help="HF model adı")
    ap.add_argument("--hf-token", type=str, default=None, help="HF token (boşsa env HF_TOKEN)")
    ap.add_argument("--min-spk", type=int, default=None, help="min_speakers")
    ap.add_argument("--max-spk", type=int, default=None, help="max_speakers")
    ap.add_argument("--no-preview", action="store_true", help="Wave preview JSON üretme")
    args = ap.parse_args()

    wav_path = Path(args.wav_path).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV bulunamadı: {wav_path}")

    params = {"min_speakers": args.min_spk, "max_speakers": args.max_spk}

    print(f"[INFO] Model   : {args.model}")
    print(f"[INFO] Audio   : {wav_path.name}")
    print(f"[INFO] Out dir : {out_root}")

    rttm_path = run_diarization(
        wav_path=wav_path,
        out_dir=out_root,
        model_name=args.model,
        hf_token=args.hf_token,
        params=params,
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
        }
        update_manifest(out_root, item)
        print("[OK] Manifest güncellendi.")
    except Exception as e:
        print(f"[WARN] Manifest güncellenemedi: {e}")

if __name__ == "__main__":
    main()
