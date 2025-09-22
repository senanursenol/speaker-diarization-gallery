from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import streamlit as st

def load_css(css_path: Path) -> None:
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

def human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:d}:{s:02d}"

def write_rttm(annotation, uri: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for segment, _, label in annotation.itertracks(yield_label=True):
            start = segment.start
            dur = segment.duration
            f.write(
                f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>\n"
            )

def make_wave_preview(wav_path: Path, out_root: Path, target_points: int = 4000) -> float:
    import soundfile as sf
    y, sr = sf.read(str(wav_path)) 

    if hasattr(y, "ndim") and y.ndim == 2:
        y = y.mean(axis=1)

    n = len(y)
    if n == 0:
        duration = 0.0
        mins: List[float] = [0.0]
        maxs: List[float] = [0.0]
        ts:   List[float] = [0.0]
    else:
        win = max(1, n // target_points)
        mins, maxs = [], []
        for i in range(0, n, win):
            chunk = y[i:i + win]
            mins.append(float(np.min(chunk)))
            maxs.append(float(np.max(chunk)))

        m = len(mins)
        duration = float(n) / float(sr) if sr else 0.0
        ts = (np.linspace(0.0, duration, num=m, endpoint=False)).astype(float).tolist()

    previews_dir = out_root / "data" / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    uri = wav_path.stem
    out_json: Dict[str, Any] = {
        "id": uri,
        "sr": int(sr),
        "duration_sec": round(duration, 6),
        "envelope": {"t": ts, "min": mins, "max": maxs},
    }
    (previews_dir / f"{uri}.wave.json").write_text(
        json.dumps(out_json, ensure_ascii=False), encoding="utf-8"
    )
    return duration

def update_manifest(out_root: Path, item: Dict[str, Any]) -> None:
    manifest_path = out_root / "data" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"items": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    manifest["items"].append(item)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
