import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import cheby1, freqs

from LPF import calc_order, chebyshev_g_values, scale_elements, format_value

st.set_page_config(page_title="Chebyshev LC Filter Designer", layout="wide")
st.title("Chebyshev Type I LC Low-Pass Filter Designer")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filter Specifications")

    fp_ghz = st.number_input("fp — 통과대역 주파수 (GHz)", value=1.0, min_value=0.001, step=0.1, format="%.3f")
    fs_ghz = st.number_input("fs — 저지대역 주파수 (GHz)", value=2.4, min_value=0.001, step=0.1, format="%.3f")
    Rp     = st.number_input("Rp — 통과대역 리플 (dB)",   value=0.2, min_value=0.01,  step=0.05, format="%.2f")
    Rs     = st.number_input("Rs — 저지대역 감쇠 (dB)",   value=60.0, min_value=1.0,  step=5.0,  format="%.1f")
    R0     = st.number_input("R0 — 특성 임피던스 (Ω)",    value=50.0, min_value=1.0,  step=1.0,  format="%.1f")

    st.divider()
    auto_N = st.checkbox("N 자동계산", value=True)
    fp_hz  = fp_ghz * 1e9
    fs_hz  = fs_ghz * 1e9

    if auto_N:
        try:
            N_min = calc_order(fp_hz, fs_hz, Rp, Rs)
        except Exception:
            N_min = 1
        N = st.number_input("N (자동 계산됨)", value=int(N_min), disabled=True)
        N = int(N_min)
    else:
        N = st.number_input("N (수동 입력)", value=7, min_value=1, max_value=20, step=1)
        N = int(N)

# ── Calculations ──────────────────────────────────────────────────────────────
try:
    g        = chebyshev_g_values(N, Rp)
    elements = scale_elements(g, N, fp_hz, R0)
    R_load   = R0 * g[N + 1]
    ok       = True
except Exception as e:
    st.error(f"계산 오류: {e}")
    ok = False

if ok:
    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    col1.metric("필터 차수  N", N)
    col2.metric("부하 임피던스  R_load", f"{R_load:.4f} Ω")

    st.divider()

    # ── Element table ─────────────────────────────────────────────────────────
    st.subheader("소자값 (Shunt-C / Series-L 토폴로지)")

    rows = []
    for elem_type, k, val in elements:
        unit = 'F' if elem_type == 'C' else 'H'
        rows.append({
            "소자":  f"{elem_type}{k}",
            "종류":  "병렬 커패시터" if elem_type == 'C' else "직렬 인덕터",
            "값":    format_value(val, unit),
            "값 (SI)": f"{val:.6e}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()

    # ── Plotly frequency response ─────────────────────────────────────────────
    st.subheader("주파수 응답")

    wc   = 2.0 * np.pi * fp_hz
    b, a = cheby1(N, Rp, wc, btype='low', analog=True)

    w_start = wc * 0.05
    w_end   = wc * 15
    w       = np.logspace(np.log10(w_start), np.log10(w_end), 4000)
    _, h    = freqs(b, a, worN=w)
    freq_ghz = w / (2 * np.pi * 1e9)
    mag_db   = 20 * np.log10(np.abs(h) + 1e-300)

    _, h_fp = freqs(b, a, worN=[2 * np.pi * fp_hz])
    _, h_fs = freqs(b, a, worN=[2 * np.pi * fs_hz])
    db_fp   = 20 * np.log10(abs(h_fp[0]) + 1e-300)
    db_fs   = 20 * np.log10(abs(h_fs[0]) + 1e-300)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.semilogx(freq_ghz, mag_db, linewidth=2, color='steelblue')

    ax.axvline(fp_ghz, color='gray',   linestyle='--', linewidth=1.2, label=f'fp = {fp_ghz:.3f} GHz  ({db_fp:.2f} dB)')
    ax.axvline(fs_ghz, color='red',    linestyle='--', linewidth=1.2, label=f'fs = {fs_ghz:.3f} GHz  ({db_fs:.2f} dB)')
    ax.axhline(-Rp,    color='orange', linestyle=':',  linewidth=1.2, label=f'-Rp = -{Rp} dB')
    ax.axhline(-Rs,    color='blue',   linestyle=':',  linewidth=1.2, label=f'-Rs = -{Rs} dB')

    ax.plot(fp_ghz, db_fp, 'o', color='gray', markersize=7, zorder=5)
    ax.plot(fs_ghz, db_fs, 'o', color='red',  markersize=7, zorder=5)

    ax.annotate(f'{db_fp:.2f} dB', xy=(fp_ghz, db_fp),
                xytext=(fp_ghz * 0.6, db_fp + 6),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')
    ax.annotate(f'{db_fs:.2f} dB', xy=(fs_ghz, db_fs),
                xytext=(fs_ghz * 1.3, db_fs + 10),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    x_min = freq_ghz[0]
    x_max = freq_ghz[-1]
    major_ticks = [t for t in [0.1, 0.2, 0.5, 1.0, 2.0, 2.4, 5.0, 10.0, 15.0] if x_min <= t <= x_max]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(v) for v in major_ticks], fontsize=8.5)
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_title(f'Chebyshev Type I LPF  (N={N}, Rp={Rp} dB, fp={fp_ghz:.3f} GHz)')
    ax.set_ylim(-105, 5)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ── g-values (advanced, hidden by default) ────────────────────────────────
    with st.expander("g값 (정규화 프로토타입)", expanded=False):
        g_rows = [{"인덱스": f"g[{i}]", "값": f"{g[i]:.8f}"} for i in range(N + 2)]
        st.dataframe(g_rows, use_container_width=True, hide_index=True)
