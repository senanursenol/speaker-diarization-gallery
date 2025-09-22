import time, warnings, argparse, os, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch, torchaudio
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=UserWarning)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42); random.seed(42); torch.manual_seed(42)

def read_mono_16k(path: str):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        yt = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        yt = torchaudio.functional.resample(yt, sr, 16000)
        y = yt.squeeze(0).numpy()
        sr = 16000
    return y.astype(np.float32), sr

def load_silero_vad_offline_first():
    try:
        from silero_vad import VoiceActivityDetector
        class _PkgWrapper:
            def __init__(self): self.vad = VoiceActivityDetector(sr=16000)
            def get_ts(self, wav, sr, threshold, min_speech_ms, min_sil_ms):
                return self.vad.get_speech_timestamps(
                    torch.from_numpy(wav), sampling_rate=sr, threshold=threshold,
                    min_speech_duration_ms=min_speech_ms, min_silence_duration_ms=min_sil_ms,
                )
        return _PkgWrapper()
    except Exception:
        vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        (get_speech_timestamps, _, _, _, _) = utils
        class _HubWrapper:
            def __init__(self, model, getter): self.model, self.getter = model, getter
            def get_ts(self, wav, sr, threshold, min_speech_ms, min_sil_ms):
                return self.getter(
                    torch.tensor(wav), self.model, sampling_rate=sr, threshold=threshold,
                    min_speech_duration_ms=min_speech_ms, min_silence_duration_ms=min_sil_ms,
                )
        return _HubWrapper(vad_model, get_speech_timestamps)

def vad_segments_from_file(path: str, sr=16000, threshold=0.5, min_speech=0.30, min_sil=0.10):
    wav, _ = read_mono_16k(path)
    vad = load_silero_vad_offline_first()
    ts = vad.get_ts(
        wav, sr, threshold=threshold,
        min_speech_ms=int(min_speech * 1000),
        min_sil_ms=int(min_sil * 1000),
    )
    return [(t["start"] / sr, t["end"] / sr) for t in ts]

def sample_chunks(y, sr, segs, win=0.75, hop=0.25, min_keep=0.40):
    chunks = []
    step = int(hop * sr); win_n = int(win * sr)
    for s, e in segs:
        i0, i1 = int(s * sr), int(e * sr)
        i = i0
        while i < i1:
            j = min(i + win_n, i1)
            if j - i >= int(min_keep * sr):
                chunks.append((i / sr, j / sr, y[i:j]))
            i += step if step > 0 else win_n
    return chunks

def embed_resemblyzer(chunks, device="cpu"):
    from resemblyzer import VoiceEncoder
    print(f"Loaded the voice encoder model on {device} in 0.03 seconds.")
    enc = VoiceEncoder(device=device)
    embs = [enc.embed_utterance(w.astype(np.float32)) for _, _, w in chunks]
    X = normalize(np.stack(embs, axis=0))
    times = [(s, e) for s, e, _ in chunks]
    return X, times

def imbalance_ratio(labels):
    """0 = dengeli, 1 = çok dengesiz."""
    _, cnt = np.unique(labels, return_counts=True)
    if len(cnt) < 2: return 1.0
    return (cnt.max() - cnt.min()) / max(cnt.sum(), 1)

def any_too_small(labels, min_count):
    _, cnt = np.unique(labels, return_counts=True)
    return (cnt < min_count).any()

def kmeans_cosine_with_penalties(X, kmin=2, kmax=12, min_frac=0.01, imb_pen=0.05, k_bias=0.01):
    from sklearn.metrics import silhouette_score
    n = len(X)
    upper = min(kmax, max(kmin, n - 1))
    if upper < 2: return None, None
    best = (-1e9, None, None)
    min_count = max(2, int(np.ceil(min_frac * n)))
    for k in range(kmin, upper + 1):
        labs = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
        if len(set(labs)) < 2:
            continue
        if any_too_small(labs, min_count):
            continue
        try:
            sil = silhouette_score(X, labs, metric="cosine")
        except Exception:
            continue
        imb = imbalance_ratio(labs)
        score = sil - imb_pen * imb + k_bias * np.log(k)
        if score > best[0]:
            best = (score, k, labs)
    return best[2], best[1]

def overcluster_then_merge(X, k_base_labels, merge_th=0.88, extra=4):
    n = len(X)
    if n < 4: return k_base_labels
    Xn = normalize(X)
    uniq = np.unique(k_base_labels)
    k0 = len(uniq)
    k_over = min(max(k0 + extra, 12), n - 1)
    if k_over <= k0:
        return k_base_labels
    labs_over = KMeans(n_clusters=k_over, n_init=20, random_state=42).fit_predict(Xn)
    cents = []
    for u in range(k_over):
        idx = np.where(labs_over == u)[0]
        if len(idx) == 0:
            cents.append(np.zeros(Xn.shape[1], dtype=np.float32))
        else:
            cents.append(normalize(Xn[idx].mean(axis=0, keepdims=True))[0])
    cents = np.stack(cents, axis=0)
    sim = cents @ cents.T
    parent = list(range(k_over))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for i in range(k_over):
        for j in range(i+1, k_over):
            if sim[i, j] >= merge_th:
                union(i, j)
    roots = {find(i): idx for idx, i in enumerate(sorted(set(find(x) for x in range(k_over))))}
    new_labs = np.array([roots[find(l)] for l in labs_over], dtype=int)
    if len(np.unique(new_labs)) < len(np.unique(k_base_labels)):
        return k_base_labels
    return new_labs

def median_labels(labels, frame=3):
    L = np.array(labels); n = len(L); r = max(1, frame // 2)
    out = L.copy()
    for i in range(n):
        a, b = max(0, i - r), min(n, i + r + 1)
        vals, cnt = np.unique(L[a:b], return_counts=True)
        out[i] = vals[cnt.argmax()]
    return out.tolist()

def chunks_to_segments(chunk_times, labels, bridge=0.15):
    segs = []
    cs, ce = chunk_times[0]; cur = labels[0]
    for (s, e), lab in zip(chunk_times[1:], labels[1:]):
        if lab == cur and s <= ce + bridge:
            ce = max(ce, e)
        else:
            segs.append((cs, ce, cur))
            cs, ce, cur = s, e, lab
    segs.append((cs, ce, cur))
    return segs

def merge_tiny_segments(seg_lab, min_len=0.40, gap_bridge=0.15):
    if not seg_lab: return seg_lab
    out = []
    for s, e, l in sorted(seg_lab):
        if not out:
            out.append([s, e, l]); continue
        ps, pe, pl = out[-1]
        if l == pl and s <= pe + gap_bridge:
            out[-1][1] = max(pe, e)
        else:
            if (e - s) < min_len:
                if l == pl and (s - pe) <= gap_bridge:
                    out[-1][1] = max(pe, e)
                else:
                    out.append([s, e, l])
            else:
                out.append([s, e, l])
    return [(s, e, l) for s, e, l in out]

def collapse_ABA(seg_lab, island_th=1.0):
    if len(seg_lab) < 3:
        return seg_lab
    out = seg_lab[:]
    i = 1
    while i < len(out) - 1:
        s0, e0, l0 = out[i-1]
        s1, e1, l1 = out[i]
        s2, e2, l2 = out[i+1]
        if l0 == l2 and l1 != l0:
            dur1 = e1 - s1
            if dur1 <= island_th and s1 <= e0 and s2 <= e1 + 1e-6:
                new_seg = (s0, e2, l0)
                out[i-1:i+2] = [new_seg]
                i = max(1, i-1)
                continue
        i += 1
    return out

def merge_close_same_speaker(seg_lab, gap_th=0.5):
    if not seg_lab:
        return seg_lab
    merged = []
    cs, ce, cl = seg_lab[0]
    for s, e, l in seg_lab[1:]:
        if l == cl and s - ce <= gap_th:
            ce = max(ce, e)
        else:
            merged.append((cs, ce, cl))
            cs, ce, cl = s, e, l
    merged.append((cs, ce, cl))
    return merged

def write_rttm(uri, seg_lab, path):
    with open(path, "w", encoding="utf-8") as f:
        for s, e, l in seg_lab:
            f.write(f"SPEAKER {uri} 1 {s:.3f} {e - s:.3f} <NA> <NA> spk{l:02d} <NA> <NA>\n")

def run(audio, device="cpu", vad_th=0.5, vad_min=0.30, vad_gap=0.10,
        win=0.75, hop=0.50, tiny=1.00, kmin=2, kmax=12,
        min_cluster_frac=0.01, imb_pen=0.05, k_bias=0.01,
        use_overcluster=True, over_extra=4, merge_th=0.88,
        bridge=0.50, median_frame=5):
    
    y, sr = read_mono_16k(audio)
    vad = vad_segments_from_file(audio, sr, threshold=vad_th, min_speech=vad_min, min_sil=vad_gap)
    if not vad: return {"error": "Konuşma bulunamadı."}
    chunks = sample_chunks(y, sr, vad, win=win, hop=hop, min_keep=0.40)
    if len(chunks) < 2: return {"error": "Yeterli örnek yok."}

    X, times = embed_resemblyzer(chunks, device=device)

    base_labels, k_sel = kmeans_cosine_with_penalties(
        X, kmin=kmin, kmax=kmax,
        min_frac=min_cluster_frac, imb_pen=imb_pen, k_bias=k_bias
    )
    if base_labels is None:
        return {"error": "Kümeleme başarısız (k seçilemedi)."}

    labels = base_labels
    if use_overcluster:
        labels = overcluster_then_merge(X, base_labels, merge_th=merge_th, extra=over_extra)

    labels = median_labels(labels, frame=int(median_frame))
    seg_lab = chunks_to_segments(times, labels, bridge=float(bridge))
    seg_lab = merge_tiny_segments(seg_lab, min_len=float(tiny), gap_bridge=float(bridge))
    seg_lab = collapse_ABA(seg_lab, island_th=1.0)
    seg_lab = merge_close_same_speaker(seg_lab, gap_th=float(bridge))

    uniq = sorted(set(l for _, _, l in seg_lab))
    remap = {l: i for i, l in enumerate(uniq)}
    seg_lab = [(s, e, remap[l]) for s, e, l in seg_lab]

    spk_dur = defaultdict(float)
    for s, e, l in seg_lab:
        spk_dur[l] += (e - s)

    return {
        "uri": Path(audio).stem,
        "speakers": len(uniq),
        "segments": seg_lab,
        "per_speaker": sorted(
            [{"spk": f"spk{idx:02d}", "sec": round(spk_dur[idx], 2)} for idx in range(len(uniq))],
            key=lambda r: r["spk"]
        )
    }

def main():
    ap = argparse.ArgumentParser(description="Resemblyzer diarization (anti-2-kilit) + Silero VAD + JER-dostu post-process + RTTM")
    ap.add_argument("audio", help="Girdi ses")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--vad-th", type=float, default=0.5)
    ap.add_argument("--vad-min", type=float, default=0.30)
    ap.add_argument("--vad-gap", type=float, default=0.10)
    ap.add_argument("--win", type=float, default=0.75)
    ap.add_argument("--hop", type=float, default=0.50)            
    ap.add_argument("--tiny", type=float, default=1.00)           
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--min-cluster-frac", type=float, default=0.01)
    ap.add_argument("--imb-pen", type=float, default=0.05)
    ap.add_argument("--k-bias", type=float, default=0.01)
    ap.add_argument("--no-overcluster", action="store_true", help="Overcluster→Merge devre dışı")
    ap.add_argument("--over-extra", type=int, default=4, help="Overcluster ek küme sayısı")
    ap.add_argument("--merge-th", type=float, default=0.88, help="Merkez benzerliği eşik (0.85–0.92 iyi)")
    ap.add_argument("--bridge", type=float, default=0.50, help="Aynı konuşmacı pencereleri arasında köprü (s)")
    ap.add_argument("--median-frame", type=int, default=5, help="Median filtresi pencere boyu (3/5/7)")
    ap.add_argument("--save-rttm", default=None, help="Hipotez RTTM yolu (varsayılan: <audio>.resemblyzer.rttm)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = run(
        args.audio, device=args.device,
        vad_th=args.vad_th, vad_min=args.vad_min, vad_gap=args.vad_gap,
        win=args.win, hop=args.hop, tiny=args.tiny,
        kmin=args.kmin, kmax=args.kmax,
        min_cluster_frac=args.min_cluster_frac,
        imb_pen=args.imb_pen, k_bias=args.k_bias,
        use_overcluster=not args.no_overcluster,
        over_extra=args.over_extra, merge_th=args.merge_th,
        bridge=args.bridge, median_frame=args.median_frame
    )
    elapsed = time.perf_counter() - t0

    if "error" in out:
        print(out["error"]); return

    uri = out["uri"]; seg_lab = out["segments"]

    print(f"Dosya (URI): {uri}")
    print(f"Bulunan konuşmacı sayısı: {out['speakers']}")
    for row in out["per_speaker"]:
        print(f"  {row['spk']}: {row['sec']} sn")
    print(f"Toplam segment sayısı: {len(seg_lab)}")
    print(f"Çalışma süresi: {elapsed:.2f} sn (device={args.device})")

    rttm_path = args.save_rttm or f"{uri}.resemblyzer.rttm"
    write_rttm(uri, seg_lab, rttm_path)
    print(f"RTTM kaydedildi: {rttm_path}")

if __name__ == "__main__":
    main()
