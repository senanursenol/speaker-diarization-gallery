from __future__ import annotations
from pathlib import Path
import json
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict
from statistics import mean
import numpy as np
import plotly.express as px

from helpers import load_css

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Set2

PLOT_BG = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_color="#0F172A"
)

BASE = Path(__file__).resolve().parent
DATA_DIR = (BASE / "data") if (BASE / "data").exists() else (BASE.parent / "data")
AUDIO_DIRS = [BASE / "audio", DATA_DIR / "audio", BASE.parent / "audio"]

st.set_page_config(page_title="Diarization Galerisi", layout="wide")
load_css(BASE / "styles.css")

def fmt_time(sec: float) -> str:
    total = int(round(sec or 0))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

@st.cache_data
def load_manifest():
    p = DATA_DIR / "manifest.json"
    if not p.exists():
        return {"items": []}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_segments(rec_id: str):
    p = DATA_DIR / "summaries" / f"{rec_id}.segments.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_wave_preview(rec_id: str):
    p = DATA_DIR / "previews" / f"{rec_id}.wave.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_derjer(rec_id: str):
    p = DATA_DIR / "metrics" / f"{rec_id}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def find_audio_path(rec_id: str):
    for d in AUDIO_DIRS:
        p = d / f"{rec_id}.wav"
        if p.exists():
            return p
    return None

def speaker_palette(speakers):
    base = ["#6EA8FE","#FFB3C1","#8CE99A","#FFD8A8","#B197FC",
            "#66D9E8","#FEC89A","#B2F2BB","#C0EB75","#F783AC"]
    return {spk: base[i % len(base)] for i, spk in enumerate(sorted(speakers))}

def _overlap_intervals(segments):
    events = []
    for s in segments:
        events.append((float(s["start"]), 1))
        events.append((float(s["end"]), -1))
    events.sort()
    active = 0; start = None; out = []
    for t, delta in events:
        prev = active
        active += delta
        if prev < 2 and active >= 2:
            start = t
        elif prev >= 2 and active < 2 and start is not None:
            out.append((start, t))
            start = None
    return out

def build_timeline_figure(rec_id: str, visible_speakers=None):
    wave = load_wave_preview(rec_id); seg = load_segments(rec_id)
    fig = go.Figure()
    if not wave or not seg:
        fig.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=40))
        return fig

    t = wave["envelope"]["t"]; vmin = wave["envelope"]["min"]; vmax = wave["envelope"]["max"]
    duration = float(wave.get("duration_sec", t[-1] if t else 0.0))
    segments = seg.get("segments", [])
    all_spk = sorted({s["speaker"] for s in segments})
    if visible_speakers is None: visible_speakers = all_spk

    ymin = float(min(vmin)) - 0.05 if vmin else -1.0
    ymax = float(max(vmax)) + 0.05 if vmax else 1.0

    fig.add_trace(go.Scatter(x=t, y=vmax, name="Waveform", mode="lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=t, y=vmin, mode="lines", fill="tonexty", line=dict(width=1), showlegend=False))

    colors = speaker_palette(all_spk)
    for s in segments:
        spk = s["speaker"]
        if spk not in visible_speakers: 
            continue
        x0, x1 = float(s["start"]), float(s["end"])
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=ymin, y1=ymax,
            fillcolor=colors[spk], opacity=0.28, line_width=0, layer="below",
        )

    ovl = _overlap_intervals(segments)
    for (x0, x1) in ovl:
        fig.add_vrect(x0=x0, x1=x1, fillcolor="#ef4444", opacity=0.12, line_width=0, layer="below")

    for s in segments:
        spk = s["speaker"]
        if spk not in visible_speakers:
            continue
        x0, x1 = float(s["start"]), float(s["end"])
        cx = 0.5 * (x0 + x1)
        dur = x1 - x0

        fig.add_trace(go.Scatter(
            x=[cx], y=[ymax], mode="markers",
            marker=dict(size=8, color=colors[spk], opacity=0),
            name=spk,
            hovertemplate=(
                f"<b>{spk}</b><br>"
                "Başlangıç: %{customdata[0]:.3f}s<br>"
                "Bitiş: %{customdata[1]:.3f}s<br>"
                "Süre: %{customdata[2]:.3f}s"
            ),
            customdata=[[x0, x1, dur]],
            showlegend=False
        ))

    for spk in all_spk:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=10, color=colors[spk]), name=spk))

    fig.update_layout(
        height=360, margin=dict(l=20,r=20,t=30,b=40),
        xaxis=dict(title="Zaman (s)", range=[0, duration], rangeslider=dict(visible=True)),
        yaxis=dict(visible=False, range=[ymin, ymax]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        **PLOT_BG
    )
    return fig

def speaker_durations(segments):
    d = defaultdict(float)
    for s in segments:
        d[s["speaker"]] += float(s["end"]) - float(s["start"])
    return sorted(d.items(), key=lambda x: x[1], reverse=True)

def build_speaker_bar(segments):
    pairs = speaker_durations(segments)
    if not pairs:
        return go.Figure()
    spk = [p[0] for p in pairs]; secs = [p[1] for p in pairs]
    colors = speaker_palette(spk)
    fig = go.Figure(go.Bar(x=spk, y=secs, marker_color=[colors[s] for s in spk]))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=40),
                      xaxis_title="Konuşmacı", yaxis_title="Süre (sn)", **PLOT_BG)
    return fig

def build_segment_hist(segments):
    durs = [float(s["end"]) - float(s["start"]) for s in segments]
    spk  = [s["speaker"] for s in segments]
    if not durs:
        return go.Figure()
    df = dict(duration=durs, speaker=spk)
    fig = px.violin(df, x="speaker", y="duration", box=True, points=False, color="speaker",
                    color_discrete_map=speaker_palette(sorted(set(spk))))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=40),
                      xaxis_title="Konuşmacı", yaxis_title="Segment süresi (sn)",
                      showlegend=False, **PLOT_BG)
    return fig

def build_activity_heatmap(segments, bin_sec=60, selected=None):
    if not segments:
        return go.Figure()
    if selected:
        segments = [s for s in segments if s["speaker"] in selected]
    speakers = sorted({s["speaker"] for s in segments})
    if not speakers:
        return go.Figure()

    dur = max(float(s["end"]) for s in segments)
    nbin = int(np.ceil(dur / bin_sec))
    mat = {spk: np.zeros(nbin, dtype=float) for spk in speakers}
    for s in segments:
        spk = s["speaker"]; a = float(s["start"]); b = float(s["end"])
        i0, i1 = int(a // bin_sec), int((b-1e-9) // bin_sec)
        for i in range(i0, i1+1):
            left, right = i*bin_sec, (i+1)*bin_sec
            mat[spk][i] += max(0.0, min(b, right) - max(a, left))

    z = np.vstack([mat[spk] for spk in speakers])
    fig = go.Figure(data=go.Heatmap(z=z, x=list(range(nbin)), y=speakers,
                                    colorscale="Teal", colorbar=dict(title="sn")))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=40),
                      xaxis_title=f"Zaman bin (×{bin_sec}s)", yaxis_title="Konuşmacı",
                      **PLOT_BG)
    return fig

def build_gauge(value, title):
    v = _clamp01(value)  
    if v is None:
        return go.Figure()
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix":"%", "font":{"size":22}},
        title={"text":title, "font":{"size":14}},
        gauge={
            "axis": {"range":[0, 100]},
            "bar":  {"color": _band_color(v)},
            "steps": [
                {"range":[0, 5],   "color":"rgba(16,185,129,.18)"},
                {"range":[5, 15],  "color":"rgba(245,158,11,.18)"},
                {"range":[15,100], "color":"rgba(239,68,68,.18)"},
            ],
            "threshold": {"line":{"color":_band_color(v), "width":3}, "value":v}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=10), **PLOT_BG)
    return fig

def build_cohort_scatter(rows, cur_id, der, jer):
    xs, ys, ids = [], [], []
    for r in rows:
        dv = _clamp01(r.get("der"))
        jv = _clamp01(r.get("jer"))
        if dv is not None and jv is not None:
            xs.append(dv); ys.append(jv); ids.append(r.get("id"))

    fig = go.Figure()
    # diğer kayıtlar
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=6, color="#94a3b8"),
        name="Diğer kayıtlar",
        hovertext=ids, hoverinfo="text+x+y"
    ))
    # bu kayıt
    dv = _clamp01(der); jv = _clamp01(jer)
    if dv is not None and jv is not None:
        fig.add_trace(go.Scatter(
            x=[dv], y=[jv], mode="markers",
            marker=dict(size=14, symbol="star", color="#2563eb",
                        line=dict(width=1,color="#1e40af")),
            name="Bu kayıt", hovertext=[cur_id], hoverinfo="text+x+y"
        ))

    fig.update_layout(
        height=260, margin=dict(l=20,r=20,t=30,b=40),
        xaxis_title="DER (%)", yaxis_title="JER (%)",
        **PLOT_BG
    )
    return fig

@st.cache_data
def load_all_derjer():
    base = DATA_DIR / "metrics"
    rows = []
    if base.exists():
        for p in sorted(base.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                rows.append({"id": p.stem, "der": d.get("der"), "jer": d.get("jer")})
            except Exception:
                pass
    return rows

def cohort_stats():
    rows = load_all_derjer()
    ders = [float(r["der"]) for r in rows if r["der"] is not None]
    jers = [float(r["jer"]) for r in rows if r["jer"] is not None]
    return {
        "der_mean": (mean(ders) if ders else None),  # 0–1 aralığında kalsın
        "jer_mean": (mean(jers) if jers else None),
        "count": len(rows)
    }

def _clamp01(x):
    try:
        v = float(x) * 100.0
        return max(0.0, min(100.0, v))
    except Exception:
        return None

def _band_color(v):
    if v is None: return "#94a3b8"
    if v <= 5:    return "#10b981"
    if v <= 15:   return "#f59e0b"
    return "#ef4444"


def render_gallery(items):
    st.markdown('<div class="gallery">', unsafe_allow_html=True)

    cols_per_row = 3
    for i in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row, gap="large")
        for col, item in zip(cols, items[i:i+cols_per_row]):
            with col:
                idx = int(item.get("index", 0))
                title = item.get("title") or f"Ses Kaydı {idx}"
                nspk  = int(item.get("n_speakers", 0))
                dur   = fmt_time(item.get("duration_sec", 0))
                key   = f"card-{item['id']}"
                label = f"{title}\nKonuşmacı sayısı: {nspk}\nSüre: {dur}"

                clicked = st.button(label, key=key, use_container_width=True)
                if clicked:
                    st.session_state["view"] = "detail"
                    st.session_state["selected_id"] = item["id"]

    st.markdown('</div>', unsafe_allow_html=True)      

def render_detail(rec_id: str, manifest_items):
    item = next((x for x in manifest_items if x.get("id") == rec_id), None)
    if not item:
        st.warning("Kayıt manifestte bulunamadı.")
        return

    # geri butonu
    st.markdown('<div class="back-wrap"></div>', unsafe_allow_html=True)
    if st.button("Galeriye dön", key="btn_back_to_gallery_sb"):
        st.session_state["view"] = "gallery"
        st.session_state.pop("selected_id", None)
        st.rerun()

    idx = item.get("index")
    title = item.get("title") or (f"Ses kaydı {idx}" if idx is not None else "Ses kaydı")
    duration = item.get("duration_sec") or 0
    try:
        n_spk = int(item.get("n_speakers") or 0)
    except (TypeError, ValueError):
        n_spk = 0

    st.subheader(title)
    st.caption(f"{fmt_time(duration)} • {n_spk} konuşmacı")

    audio_path = find_audio_path(rec_id)
    if audio_path:
        st.audio(str(audio_path))
        st.caption(f"Dosya: {audio_path.name} • {fmt_time(duration)}")
    else:
        st.info(f"Ses bulunamadı: {rec_id}.wav")

    seg = load_segments(rec_id); segments = seg.get("segments", []) if seg else []
    speakers_all = sorted({s["speaker"] for s in segments})
    selected = st.multiselect("Konuşmacı filtresi", speakers_all, default=speakers_all) if speakers_all else []

    wave = load_wave_preview(rec_id)
    if wave and segments:
        st.write("Zaman ekseni (waveform + konuşmacı segmentleri)")
        fig = build_timeline_figure(rec_id, visible_speakers=selected if selected else speakers_all)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waveform önizlemesi veya segment bilgisi bulunamadı.")

    if segments:
        st.markdown("### Analiz")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Konuşmacı toplam süreleri")
            st.plotly_chart(
                build_speaker_bar([s for s in segments if (not selected or s['speaker'] in selected)]),
                use_container_width=True
            )
        with c2:
            st.caption("Segment uzunluğu dağılımı")
            st.plotly_chart(
                build_segment_hist([s for s in segments if (not selected or s['speaker'] in selected)]),
                use_container_width=True
            )

        st.caption("Zaman-yoğunluk ısı haritası")
        st.plotly_chart(
            build_activity_heatmap(
                [s for s in segments if (not selected or s['speaker'] in selected)],
                bin_sec=60, selected=selected if selected else None
            ),
            use_container_width=True, theme=None
        )

        rows = [{"start": round(s["start"],3), "end": round(s["end"],3), "speaker": s["speaker"]}
                for s in segments if (not selected or s["speaker"] in selected)]
        st.write("Segmentler")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    metrics = load_derjer(rec_id)
    if metrics:
        der = metrics.get("der")
        jer = metrics.get("jer")
        cohort = load_all_derjer()

        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            st.plotly_chart(build_gauge(der, "DER"), use_container_width=True)
        with c2:
            st.plotly_chart(build_gauge(jer, "JER"), use_container_width=True)
        with c3:
            st.plotly_chart(build_cohort_scatter(cohort, rec_id, der, jer), use_container_width=True)

        note = metrics.get("notes")
        if note:
            st.markdown(f'<div class="metric-note">ℹ︎ {note}</div>', unsafe_allow_html=True)
    else:
        st.info("Bu kayıt için DER/JER metrik dosyası bulunamadı.")

def main():
    st.markdown("""
        <div class="hero">
            <div class="hero-badge">
                <h1>DIARIZATION GALERİSİ</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)

    manifest = load_manifest()
    items = list(sorted(manifest.get("items", []), key=lambda x: x.get("index", 0)))
    if not items:
        st.warning("Henüz manifest bulunamadı.")
        return

    if "view" not in st.session_state: 
        st.session_state["view"] = "gallery"
    if "selected_id" not in st.session_state and items: 
        st.session_state["selected_id"] = items[0]["id"]

    st.markdown('<div id="gallery-start"></div>', unsafe_allow_html=True)

    if st.session_state["view"] == "gallery":
        render_gallery(items)
    else:
        render_detail(st.session_state["selected_id"], items)

if __name__ == "__main__":
    main()
