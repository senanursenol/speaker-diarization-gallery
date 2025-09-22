import argparse, os, time
from pathlib import Path
from collections import OrderedDict, defaultdict
import numpy as np
import librosa
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from speechbrain.inference import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

SR = 16000
EMB_WIN_S = 1.5
MIN_K, MAX_K = 3, 8
PP_MIN_SPK_TOTAL = 2.0
PP_MIN_SEG_S = 0.6

CLUSTERER = "spec"  
EMB_HOP_S = 0.25
VIT_PEN   = 0.35
BRIDGE_S  = 0.10

def pct(x): return f"{x*100:.2f}%"

def load_rttm(path: Path) -> Annotation:
    ann = Annotation(uri=path.stem)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if not p or p[0] != "SPEAKER": continue
            start = float(p[3]); dur = float(p[4]); spk = p[7]
            ann[Segment(start, start+dur)] = spk
    return ann

_SILERO = {"model": None, "utils": None}
def get_silero():
    if _SILERO["model"] is None:
        m, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        _SILERO["model"] = m
        _SILERO["utils"] = utils
    return _SILERO["model"], _SILERO["utils"]

def silero_vad_segments(wav, sr, min_seg_ms=400, thresh=0.6):
    try:
        model, utils = get_silero()
        (get_speech_timestamps, _, _, _, _) = utils
        wav_t = torch.from_numpy(wav).float()
        if wav_t.ndim == 1: wav_t = wav_t.unsqueeze(0)
        ts = get_speech_timestamps(wav_t, model, sampling_rate=sr,
                                   threshold=thresh, min_speech_duration_ms=min_seg_ms)
        return [(int(t['start']), int(t['end'])) for t in ts]
    except Exception:
        return None

def energy_vad_segments(wav, sr, top_db=30, frame_ms=20, hop_ms=10, min_seg_ms=400):
    intervals = librosa.effects.split(
        wav, top_db=top_db,
        frame_length=int(sr*frame_ms/1000),
        hop_length=int(sr*hop_ms/1000),
    )
    min_len = int(sr*(min_seg_ms/1000))
    return [(int(s), int(e)) for s,e in intervals if e-s >= min_len]

def pad_to_length(x, tgt_len):
    if len(x) >= tgt_len: return x[:tgt_len]
    out = np.zeros(tgt_len, dtype=x.dtype); out[:len(x)] = x; return out

def segment_embeddings(wav, sr, segs, encoder, emb_win_s, emb_hop_s):
    out = []
    win = int(sr*emb_win_s); hop = int(sr*emb_hop_s)
    with torch.no_grad():
        for s, e in segs:
            i = s
            if e - s < win:
                chunk = pad_to_length(wav[s:e], win)
                wav_tensor = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
                wav_lens = torch.tensor([1.0], dtype=torch.float32)
                emb = encoder.encode_batch(wav_tensor, wav_lens).squeeze().cpu().numpy()
                emb = emb / max(np.linalg.norm(emb), 1e-9)
                out.append((s/sr, emb_win_s, emb)); continue
            while i + win <= e:
                chunk = wav[i:i+win]
                wav_tensor = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
                wav_lens = torch.tensor([1.0], dtype=torch.float32)
                emb = encoder.encode_batch(wav_tensor, wav_lens).squeeze().cpu().numpy()
                emb = emb / max(np.linalg.norm(emb), 1e-9)
                out.append((i/sr, emb_win_s, emb))
                i += hop
    return out

def extract_vad_and_embs(wav_path: Path, encoder):
    wav, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    segs = silero_vad_segments(wav, sr, min_seg_ms=400, thresh=0.60)
    if not segs:
        segs = energy_vad_segments(wav, sr, min_seg_ms=400)
    if not segs: return None, None
    embs = segment_embeddings(wav, sr, segs, encoder, emb_win_s=EMB_WIN_S, emb_hop_s=EMB_HOP_S)
    if not embs: return None, None
    X = np.stack([e[2] for e in embs], axis=0)
    times = [(st, dur) for (st, dur, _) in embs]
    return X, times

def _agglom_fit_predict(X, k):
    if X.shape[0] <= k:
        k = max(2, min(MIN_K, X.shape[0]-1))
    try:
        clus = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    except TypeError:
        clus = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
    return clus.fit_predict(X)

def _affinity_cosine(Xn, topk=None):
    A = np.clip(Xn @ Xn.T, 0.0, 1.0)
    np.fill_diagonal(A, 1.0)
    if topk is not None and topk < A.shape[0]:
        idx = np.argpartition(-A, kth=topk, axis=1)[:, :topk]
        mask = np.zeros_like(A, dtype=bool)
        rows = np.arange(A.shape[0])[:, None]
        mask[rows, idx] = True
        A = np.where(mask, A, 0.0)
    A = 0.5*(A + A.T)
    return A

def _spectral_embed(A, k):
    d = np.clip(A.sum(axis=1), 1e-9, None)
    Dm12 = np.diag(1.0/np.sqrt(d))
    L = Dm12 @ A @ Dm12
    w, V = np.linalg.eigh(L)
    idx = np.argsort(w)[-k:]
    U = V[:, idx]
    U = U / np.clip(np.linalg.norm(U, axis=1, keepdims=True), 1e-9, None)
    return U

def spectral_labels(X, k, topk_frac=0.1):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    N = Xn.shape[0]
    topk = max(10, int(topk_frac*N))
    A = _affinity_cosine(Xn, topk=topk)
    U = _spectral_embed(A, k)
    labs = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(U)
    return labs

def choose_k_spectral(X, min_k=MIN_K, max_k=MAX_K):
    best_k, best_score = None, -1e9
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    N = Xn.shape[0]
    topk = max(10, int(0.1*N))
    A = _affinity_cosine(Xn, topk=topk)
    cache_U = {}
    for k in range(min_k, min(max_k, N-1)+1):
        try:
            if k not in cache_U:
                U = _spectral_embed(A, k)
                cache_U[k] = U
            else:
                U = cache_U[k]
            labs = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(U)
            sil = silhouette_score(U, labs, metric="cosine")
            if sil > best_score:
                best_k, best_score = k, sil
        except Exception:
            continue
    return best_k or min_k

def centroid_reassign(X, labels):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    uniq = sorted(set(labels))
    cents = []
    for u in uniq:
        idx = np.where(labels == u)[0]
        c = Xn[idx].mean(axis=0)
        c = c / max(np.linalg.norm(c), 1e-9)
        cents.append(c)
    cents = np.stack(cents, axis=0)
    sims = Xn @ cents.T
    new = sims.argmax(axis=1)
    remap = {u: i for i, u in enumerate(uniq)}
    old = np.array([remap[l] for l in labels])
    return new if (new != old).any() else old

def median_labels(labels, frame=5):
    L = np.array(labels); n = len(L); r = max(1, frame//2)
    out = L.copy()
    for i in range(n):
        a,b = max(0, i-r), min(n, i+r+1)
        vals, cnt = np.unique(L[a:b], return_counts=True)
        out[i] = vals[cnt.argmax()]
    return out

def viterbi_resegment(X, labels, switch_pen=VIT_PEN):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    uniq = sorted(set(labels))
    cents = []
    for u in uniq:
        idx = np.where(labels == u)[0]
        c = Xn[idx].mean(axis=0); c /= max(np.linalg.norm(c), 1e-9)
        cents.append(c)
    C = np.stack(cents, axis=0)
    emit = Xn @ C.T
    N,K = emit.shape
    dp = np.full((N,K), -1e9, dtype=np.float32)
    bp = np.full((N,K), -1, dtype=np.int32)
    dp[0] = emit[0]
    for t in range(1,N):
        best_prev = dp[t-1].max()
        best_idx  = dp[t-1].argmax()
        for k in range(K):
            stay = dp[t-1,k]
            switch = best_prev - switch_pen
            if stay >= switch:
                dp[t,k] = stay + emit[t,k]; bp[t,k] = k
            else:
                dp[t,k] = switch + emit[t,k]; bp[t,k] = best_idx
    path = np.empty(N, dtype=np.int32)
    path[-1] = int(dp[-1].argmax())
    for t in range(N-2,-1,-1):
        path[t] = bp[t+1, path[t+1]]
    remap = {i:u for i,u in enumerate(uniq)}
    return np.array([remap[i] for i in path], dtype=int)

def weld_segments(times, labels, bridge=BRIDGE_S):
    if not times: return []
    segs=[]; cur_lab=int(labels[0])
    cur_start=float(times[0][0]); cur_end=float(times[0][0]+times[0][1])
    for (st,dur), lab in zip(times[1:], labels[1:]):
        st=float(st); en=float(st+dur); lab=int(lab)
        if lab==cur_lab and st <= cur_end + bridge:
            cur_end = max(cur_end, en)
        else:
            segs.append((cur_start, cur_end-cur_start, cur_lab))
            cur_start, cur_end, cur_lab = st, en, lab
    segs.append((cur_start, cur_end-cur_start, cur_lab))
    return segs

def collapse_ABA(seg_lab, island_th=0.8):
    if len(seg_lab)<3: return seg_lab
    out = seg_lab[:]; i=1
    while i < len(out)-1:
        s0,d0,l0 = out[i-1]; s1,d1,l1 = out[i]; s2,d2,l2 = out[i+1]
        e0=s0+d0; e1=s1+d1; e2=s2+d2
        if l0==l2 and l1!=l0 and d1<=island_th and s1<=e0 and s2<=e1+1e-6:
            out[i-1:i+2] = [(s0, e2-s0, l0)]; i = max(1, i-1); continue
        i+=1
    return out

def pp_filter(segs, min_seg=PP_MIN_SEG_S, min_spk_total=PP_MIN_SPK_TOTAL):
    merged=[]
    for s,d,l in segs:
        if merged and merged[-1][2]==l and (s-(merged[-1][0]+merged[-1][1]))<=0.2:
            ps,pd,pl = merged[-1]; merged[-1] = (ps, (s+d)-ps, pl)
        else:
            if d<min_seg and merged and merged[-1][2]==l:
                ps,pd,pl = merged[-1]; merged[-1]=(ps,(s+d)-ps,pl)
            else:
                merged.append((s,d,l))
    tot=defaultdict(float)
    for s,d,l in merged: tot[l]+=d
    return [(s,d,l) for (s,d,l) in merged if tot[l]>=min_spk_total]

def remap_labels_in_order_str(segments_sdli):
    order = OrderedDict()
    out=[]
    for s,d,l in segments_sdli:
        if l not in order:
            order[l] = f"spk{len(order):02d}"
        out.append((s,d,order[l]))
    return out

def write_rttm_from_sd(segments_sdspk, uri, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s,d,spk in segments_sdspk:
            f.write(f"SPEAKER {uri} 1 {s:.3f} {d:.3f} <NA> <NA> {spk} <NA> <NA>\n")

def diarize_from_embs(X, times):
    if X is None or len(X) < 4: return []
    if CLUSTERER == "spec":
        k = choose_k_spectral(X, min_k=MIN_K, max_k=MAX_K)
        labels = spectral_labels(X, k)
    else:
        best_k, best_s, best_lab = None, -1e9, None
        for kk in range(MIN_K, MAX_K+1):
            if X.shape[0] <= kk: continue
            try:
                labs = _agglom_fit_predict(X, kk)
                sil = silhouette_score(X, labs, metric="cosine")
                if sil > best_s: best_k, best_s, best_lab = kk, sil, labs
            except Exception:
                continue
        k = best_k or max(2, min(MIN_K, X.shape[0]-1))
        labels = best_lab if best_lab is not None else _agglom_fit_predict(X, k)
    labels = centroid_reassign(X, labels)
    labels = median_labels(labels, frame=5)
    labels = viterbi_resegment(X, labels, switch_pen=VIT_PEN)
    segs_w = weld_segments(times, labels, bridge=BRIDGE_S)
    segs_w = collapse_ABA(segs_w, island_th=0.8)
    segs_w = pp_filter(segs_w, min_seg=PP_MIN_SEG_S, min_spk_total=PP_MIN_SPK_TOTAL)
    return segs_w

def process_one(wav_path: Path, gt_path: Path, encoder, outdir: Path):
    X, times = extract_vad_and_embs(wav_path, encoder)
    if X is None:
        print(f"[SKIP] VAD/emb yok: {wav_path.name}")
        return None
    segs = diarize_from_embs(X, times)
    if not segs:
        print(f"[SKIP] Sonuç yok: {wav_path.name}")
        return None
    der_metric = DiarizationErrorRate()
    jer_metric = JaccardErrorRate()
    ref = load_rttm(gt_path)
    hyp = Annotation(uri=wav_path.stem)
    for s,d,l in segs: hyp[Segment(s, s+d)] = f"spk{int(l):02d}"
    der = der_metric(ref, hyp); jer = jer_metric(ref, hyp)

    outdir.mkdir(parents=True, exist_ok=True)
    segs_lbl = remap_labels_in_order_str(segs)
    best_path = outdir / f"{wav_path.stem}.sb.rttm"
    write_rttm_from_sd(segs_lbl, wav_path.stem, best_path)

    spkdur = defaultdict(float)
    for s,d,l in segs: spkdur[int(l)] += d
    uniq = sorted(spkdur)
    print(f"\nDosya: {wav_path.stem}")
    print(f"SB  -> spk: {len(uniq)}, seg: {len(segs)}")
    for u in uniq: print(f"  SB   spk{u:02d}: {spkdur[u]:.2f} sn")
    print(f"Skor -> DER: {pct(der)} | JER: {pct(jer)}")
    print(f"[RTTM] yazıldı -> {best_path.name}")
    return (wav_path.stem, der, jer)

def main():
    ap = argparse.ArgumentParser(description="SB (sabit en iyi ayar) — tek dosya/klasör, RTTM yaz + DER/JER raporla")
    # tek dosya
    ap.add_argument("wav", nargs="?", help="Girdi WAV")
    ap.add_argument("gt_rttm", nargs="?", help="Ground-truth RTTM")
    # klasör modu
    ap.add_argument("--indir", help="WAV klasörü")
    ap.add_argument("--pattern", default="*.wav", help="Klasör modunda desen (vars: *.wav)")
    ap.add_argument("--gt-dir", default="pred", help="GT RTTM klasörü (vars: pred)")
    # genel
    ap.add_argument("--outdir", default="pred", help="RTTM çıkış klasörü (vars: pred)")
    args = ap.parse_args()

    savedir = Path("pretrained_models/sb_ecapa"); savedir.mkdir(parents=True, exist_ok=True)
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir), local_strategy=LocalStrategy.COPY,
    )

    if args.wav and args.gt_rttm and not args.indir:
        process_one(Path(args.wav), Path(args.gt_rttm), encoder, Path(args.outdir))
        return

    if not args.indir:
        raise SystemExit("Tek dosya vermediysen --indir kullan.")

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    gtdir = Path(args.gt_dir)
    wavs = sorted(indir.glob(args.pattern))
    if not wavs:
        print(f"Hiç WAV bulunamadı: {indir}/{args.pattern}"); return

    rows = []
    for w in wavs:
        gt = gtdir / f"{w.stem}.rttm"
        if not gt.exists():
            print(f"[SKIP] GT yok: {gt.name}")
            continue
        res = process_one(w, gt, encoder, outdir)
        if res: rows.append(res)

    if not rows:
        print("Değerlendirilecek dosya yok."); return

    print("\n--- Çoklu Dosya DER/JER (SB, yüzde) ---")
    print(f"{'Audio':<10} {'DER':>7} {'JER':>7}")
    for stem, der, jer in sorted(rows, key=lambda x: x[0]):
        print(f"{stem:<10} {der*100:6.2f}% {jer*100:6.2f}%")
    avg_der = sum(r[1] for r in rows) / len(rows)
    avg_jer = sum(r[2] for r in rows) / len(rows)
    print(f"\nOrtalama: DER={avg_der*100:.2f}% | JER={avg_jer*100:.2f}%")

if __name__ == "__main__":
    main()
