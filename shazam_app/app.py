"""
Sonic Signatures: 'Zapp tain America'
EE200: Signals, Systems and Networks | IIT Kanpur
Streamlit app: Single-clip mode + Batch mode
"""

import os, io, pickle, tempfile, glob
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as scipy_signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter
from collections import defaultdict
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sonic Signatures — EE200",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SR        = 22050
N_FFT     = 2048
HOP       = 512
FAN       = 15
FREQ_BITS = 10
DT_BITS   = 10
MAX_DT    = 200
AMP_MIN   = -60
F_MAX     = 5000
NEIGH     = 20

DB_PATH   = os.path.join(os.path.dirname(__file__), "fingerprint_db.pkl")
SONGS_DIR = os.path.join(os.path.dirname(__file__), "songs")

# ─────────────────────────────────────────────────────────────────────────────
# Core DSP functions
# ─────────────────────────────────────────────────────────────────────────────
def load_audio_bytes(data: bytes) -> tuple:
    """Load audio from raw bytes (wav). Returns (y_float32, sr)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    sr, raw = wavfile.read(tmp_path)
    os.unlink(tmp_path)
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    y = raw.astype(np.float32)
    y /= (np.max(np.abs(y)) + 1e-9)
    # Resample to SR if needed (simple decimation/interpolation)
    if sr != SR:
        n_new = int(len(y) * SR / sr)
        y = np.interp(np.linspace(0, len(y)-1, n_new),
                      np.arange(len(y)), y).astype(np.float32)
    MAX_SECONDS = 15
    if len(y) > SR * MAX_SECONDS:
        y = y[: SR * MAX_SECONDS]
    return y, SR


def compute_spectrogram(y, sr=SR, n_fft=N_FFT, hop=HOP):
    f, t, Zxx = scipy_signal.stft(y, fs=sr, window='hann',
                                   nperseg=n_fft, noverlap=n_fft - hop)
    Sdb = 20 * np.log10(np.abs(Zxx) + 1e-12)
    return f, t, Sdb


def extract_peaks(Sdb, f, t):
    f_mask = f <= F_MAX
    S = Sdb[f_mask, :]
    struct = np.ones((NEIGH, NEIGH), dtype=bool)
    S_max = maximum_filter(S, footprint=struct, mode='constant', cval=S.min())
    local_max = (S == S_max) & (S > AMP_MIN)
    fi_arr, ti_arr = np.where(local_max)
    peaks = list(zip(ti_arr.tolist(), fi_arr.tolist()))
    by_frame = defaultdict(list)
    for ti, fi in peaks:
        by_frame[ti].append((S[fi, ti], ti, fi))
    out = []
    for ti, cands in by_frame.items():
        cands.sort(key=lambda x: -x[0])
        for _, ti2, fi in cands[:8]:
            out.append((ti2, fi))
    return out, f_mask


def generate_hashes(peaks):
    peaks_sorted = sorted(peaks, key=lambda x: x[0])
    result = []
    for i, (t1, f1) in enumerate(peaks_sorted):
        for j in range(1, FAN + 1):
            if i + j >= len(peaks_sorted):
                break
            t2, f2 = peaks_sorted[i + j]
            dt = t2 - t1
            if 0 <= dt <= MAX_DT:
                h = (int(f1) << (DT_BITS + FREQ_BITS)) | (int(f2) << DT_BITS) | int(dt)
                result.append((h, t1))
    return result


def identify(y, db) -> tuple:
    """Returns (prediction, scores_dict, matches_dict, best_offsets_dict, f, t, Sdb, peaks, f_mask)"""
    f, t, Sdb = compute_spectrogram(y)
    peaks, f_mask = extract_peaks(Sdb, f, t)
    hashes = generate_hashes(peaks)

    matches = defaultdict(list)
    for h, t_anc in hashes:
        for sname, t_db in db.get(h, []):
            matches[sname].append(t_db - t_anc)

    scores = {}
    best_offsets = {}
    for sname, offsets in matches.items():
        if not offsets:
            continue
        bins = np.arange(min(offsets), max(offsets) + 2)
        counts, bin_edges = np.histogram(offsets, bins=bins)
        scores[sname] = int(counts.max())
        best_offsets[sname] = int(bin_edges[np.argmax(counts)])

    prediction = max(scores, key=scores.get) if scores else "No match"
    return prediction, scores, dict(matches), best_offsets, f, t, Sdb, peaks, f_mask


# ─────────────────────────────────────────────────────────────────────────────
# Load database (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fingerprint database…")
def load_db():
    with open(DB_PATH, "rb") as fp:
        return pickle.load(fp)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
ACCENT = "#1a237e"

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_results(y, f, t, Sdb, peaks, f_mask, scores, best_offsets, matches, prediction):
    DISPLAY_STEP = 4
    t_plot = t[::DISPLAY_STEP]
    Sdb_plot = Sdb[f_mask, ::DISPLAY_STEP]
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Waveform ──────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    t_ax = np.arange(len(y)) / SR
    ax0.plot(t_ax, y, lw=0.4, color="steelblue")
    ax0.set_xlabel("Time (s)"); ax0.set_ylabel("Amplitude")
    ax0.set_title("Query Waveform", fontweight="bold")
    ax0.set_xlim([0, t_ax[-1]]); ax0.grid(True, alpha=0.3)

    # ── 2. Spectrogram ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.pcolormesh(t_plot, f[f_mask]/1000, Sdb_plot,
                        shading="gouraud", cmap="magma", vmin=-80, vmax=0)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Frequency (kHz)")
    ax1.set_title("Spectrogram (N=2048, hop=512)", fontweight="bold")
    plt.colorbar(im, ax=ax1, label="dB")

    # ── 3. Constellation ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    im2 = ax2.pcolormesh(t_plot, f[f_mask]/1000, Sdb_plot,
                         shading="gouraud", cmap="magma", vmin=-80, vmax=0)
    pt = [t[ti] for ti, fi in peaks if fi < np.sum(f_mask)]
    pf = [f[fi]/1000 for ti, fi in peaks if fi < np.sum(f_mask)]
    ax2.scatter(pt, pf, s=5, c="cyan", linewidths=0, zorder=3,
                label=f"{len(peaks)} peaks")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Frequency (kHz)")
    ax2.set_title(f"Constellation Map ({len(peaks)} peaks)", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8, facecolor="black", labelcolor="white")
    plt.colorbar(im2, ax=ax2, label="dB")

    # ── 4. Offset histograms ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    n_songs = len(matches)
    if n_songs == 0:
        ax3.text(0.5, 0.5, "No matches found", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=14)
    else:
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_songs, subplot_spec=gs[2, :], wspace=0.4)
        ax3.set_visible(False)
        top_matches = sorted(matches.keys(), key=lambda s: scores.get(s, 0), reverse=True)[:6]
        for idx, sname in enumerate(top_matches):
            offsets = matches[sname]
            ax_h = fig.add_subplot(inner_gs[idx])
            if not offsets:
                continue
            bins = np.arange(min(offsets), max(offsets) + 2)
            counts, _ = np.histogram(offsets, bins=bins)
            color = "#2ecc71" if sname == prediction else "#3498db"
            ax_h.bar(bins[:-1], counts, width=1, color=color, edgecolor="none")
            short = sname.replace("_", " ")
            if len(short) > 18:
                short = short[:16] + "…"
            ax_h.set_title(short, fontsize=7,
                           fontweight="bold" if sname == prediction else "normal",
                           color="green" if sname == prediction else "black")
            ax_h.set_xlabel("Offset", fontsize=6)
            ax_h.set_ylabel("Count", fontsize=6)
            ax_h.tick_params(labelsize=6)
            if sname == prediction:
                ax_h.axvline(best_offsets.get(sname, 0), color="red", lw=1.5)
                ax_h.set_facecolor("#f0fff0")

    fig.suptitle(
        f'🎵  Identified: "{prediction.replace("_", " ")}"   '
        f'(score = {scores.get(prediction, 0)})',
        fontsize=14, fontweight="bold", color=ACCENT, y=1.01
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    db = load_db()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/1/1d/IIT_Kanpur_Logo.svg/200px-IIT_Kanpur_Logo.svg.png", width=80)
        st.markdown("## 🎵 Sonic Signatures")
        st.markdown("**EE200**  \nIIT Kanpur | Dr. Tushar Sandhan")
        st.divider()
        mode = st.radio("**Mode**", ["🔍 Single-Clip", "📦 Batch"], index=0)
        st.divider()
        st.markdown("**Database**")
        song_names = sorted({name for entries in db.values() for name, _ in entries})
        for s in song_names:
            st.markdown(f"• {s.replace('_', ' ')}")
        st.divider()
        st.markdown(
            "<small>Shazam-style fingerprinting via STFT constellation "
            "and paired-hash offset voting.</small>",
            unsafe_allow_html=True
        )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,{ACCENT},#283593);
                    padding:24px 32px;border-radius:12px;margin-bottom:24px'>
          <h1 style='color:white;margin:0;font-size:2rem'>🎵 Sonic Signatures</h1>
          <p style='color:#c5cae9;margin:4px 0 0'>
            EE200 Course Project &nbsp;|&nbsp; Audio Fingerprint Identifier
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    if "Single" in mode:
        # ── SINGLE-CLIP MODE ─────────────────────────────────────────────────
        st.subheader("🔍 Single-Clip Identification")
        st.markdown(
            "Upload a **.wav** query clip (a few seconds from any of the 8 songs). "
            "The app will show the **spectrogram**, **constellation map**, "
            "**offset histograms**, and the identified song."
        )

        uploaded = st.file_uploader(
            "Upload query clip (.wav)", type=["wav"],
            help="Upload a short clip (5–30 s recommended)"
        )

        if uploaded is not None:
            if uploaded.size > 15 * 1024 * 1024:
                st.error("Please upload a WAV file smaller than 15 MB.")
                st.stop()
            data = uploaded.read()
            st.audio(data, format="audio/wav")

            with st.spinner("Fingerprinting and identifying…"):
                y, sr = load_audio_bytes(data)
                pred, scores, matches, best_offsets, f, t, Sdb, peaks, f_mask = identify(y, db)

            # Result banner
            score = scores.get(pred, 0)
            conf  = min(100, int(score / 150))
            color = "#2ecc71" if score > 100 else "#e74c3c"
            st.markdown(
                f"""
                <div style='background:{color}22;border-left:5px solid {color};
                            padding:16px 20px;border-radius:8px;margin:16px 0'>
                  <h2 style='margin:0;color:{color}'>
                    🎶 &nbsp; {pred.replace("_"," ")}
                  </h2>
                  <p style='margin:4px 0 0;color:#555'>
                    Confidence score: <b>{score}</b> aligned hashes
                    &nbsp;|&nbsp; ~{conf}% confidence
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Score table
            with st.expander("📊 All song scores", expanded=False):
                df_scores = pd.DataFrame(
                    sorted(scores.items(), key=lambda x: -x[1]),
                    columns=["Song", "Score"]
                )
                df_scores["Song"] = df_scores["Song"].str.replace("_", " ")
                df_scores["Match"] = df_scores["Song"].apply(
                    lambda s: "✅" if s == pred.replace("_"," ") else ""
                )
                st.dataframe(df_scores, width='stretch', hide_index=True)

            # Plots
            st.markdown("---")
            st.subheader("📈 Intermediate Steps")
            with st.spinner("Rendering plots…"):
                fig = plot_results(y, f, t, Sdb, peaks, f_mask,
                                   scores, best_offsets, matches, pred)
                st.pyplot(fig, width='stretch')
                plt.close(fig)

            st.info("Detailed tab plots disabled on Streamlit Cloud to reduce memory usage.")

    # ══════════════════════════════════════════════════════════════════════════
    else:
        # ── BATCH MODE ───────────────────────────────────────────────────────
        st.subheader("📦 Batch Identification")
        st.markdown(
            "Upload **multiple .wav clips** at once. "
            "The app will identify each one and let you download "
            "`results.csv` with columns `filename, prediction`."
        )
        st.info("The `prediction` column contains the matched song's filename **without extension**, "
                "exactly as required by the evaluation script.", icon="ℹ️")

        uploaded_files = st.file_uploader(
            "Upload query clips (.wav)", type=["wav"],
            accept_multiple_files=True
        )

        if uploaded_files:
            results = []
            progress = st.progress(0, text="Processing…")
            status_placeholder = st.empty()

            for i, uf in enumerate(uploaded_files):
                status_placeholder.markdown(f"**Processing:** `{uf.name}` ({i+1}/{len(uploaded_files)})")
                data = uf.read()
                y, sr = load_audio_bytes(data)
                pred, scores, _, _, _, _, _, _, _ = identify(y, db)
                results.append({
                    "filename": os.path.splitext(uf.name)[0],
                    "prediction": pred
                })
                progress.progress((i + 1) / len(uploaded_files),
                                  text=f"Processed {i+1}/{len(uploaded_files)}")

            status_placeholder.empty()
            progress.empty()

            df = pd.DataFrame(results, columns=["filename", "prediction"])

            st.success(f"✅ Identified {len(df)} clips!", icon="✅")

            # Results table
            df_display = df.copy()
            df_display["Song Identified"] = df_display["prediction"].str.replace("_", " ")
            df_display["✓"] = df_display.apply(
                lambda r: "✅" if r["filename"].replace(" ","_") in r["prediction"]
                          or r["prediction"] in r["filename"] else "❓", axis=1
            )
            st.dataframe(
                df_display[["filename", "Song Identified"]],
                width='stretch', hide_index=True
            )

            # Download button
            csv_str = df[["filename", "prediction"]].to_csv(index=False)
            st.download_button(
                label="⬇️  Download results.csv",
                data=csv_str,
                file_name="results.csv",
                mime="text/csv",
                type="primary",
                width='stretch',
            )

            st.markdown("**Preview of results.csv:**")
            st.code(csv_str, language="text")


if __name__ == "__main__":
    main()
