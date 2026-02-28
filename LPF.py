import numpy as np
from scipy.signal import cheb1ord, cheby1, freqs
import matplotlib.pyplot as plt


def calc_order(fp, fs, Rp, Rs):
    """최소 필터 차수 계산 (scipy 이용)"""
    N, _ = cheb1ord(2*np.pi*fp, 2*np.pi*fs, Rp, Rs, analog=True)
    return N


def chebyshev_g_values(N, Rp_dB):
    """
    Chebyshev Type I 정규화 프로토타입 g값 계산
    (fc = 1 rad/s, R0 = 1 Ohm 기준)

    Reference: Pozar, "Microwave Engineering" / Zverev, "Handbook of Filter Synthesis"
    """
    beta  = np.log(1.0 / np.tanh(Rp_dB * np.log(10) / 40.0))
    gamma = np.sinh(beta / (2.0 * N))

    a = [np.sin((2*k - 1) * np.pi / (2*N)) for k in range(1, N + 1)]
    b = [gamma**2 + np.sin(k * np.pi / N)**2 for k in range(1, N + 1)]

    g = np.zeros(N + 2)
    g[0] = 1.0
    g[1] = 2.0 * a[0] / gamma
    for k in range(2, N + 1):
        g[k] = (4.0 * a[k-2] * a[k-1]) / (b[k-2] * g[k-1])
    g[N + 1] = 1.0 if N % 2 == 1 else (1.0 / np.tanh(beta / 4.0))**2

    return g


def scale_elements(g, N, fp, R0):
    """
    g값 -> 실제 L, C 변환 (병렬 C -> 직렬 L -> ... 토폴로지)

    홀수 소자: 병렬 커패시터  C = g / (2pi*fp*R0)
    짝수 소자: 직렬 인덕터    L = g*R0 / (2pi*fp)
    """
    wc = 2.0 * np.pi * fp
    elements = []
    for k in range(1, N + 1):
        if k % 2 == 1:
            elements.append(('C', k, g[k] / (wc * R0)))
        else:
            elements.append(('L', k, g[k] * R0 / wc))
    return elements


def format_value(val, unit_base):
    """값을 적절한 단위로 변환"""
    if unit_base == 'F':
        if val >= 1e-3:   return f"{val*1e3:.4f} mF"
        if val >= 1e-6:   return f"{val*1e6:.4f} uF"
        if val >= 1e-9:   return f"{val*1e9:.4f} nF"
        return                   f"{val*1e12:.4f} pF"
    else:  # H
        if val >= 1e-0:   return f"{val:.4f} H"
        if val >= 1e-3:   return f"{val*1e3:.4f} mH"
        if val >= 1e-6:   return f"{val*1e6:.4f} uH"
        return                   f"{val*1e9:.4f} nH"


def print_results(N, g, elements, fp, fs, Rp, Rs, R0):
    print("=" * 55)
    print("       Chebyshev Type I LC LPF Design")
    print("=" * 55)
    print(f"  Passband freq    fp = {fp/1e9:.2f} GHz")
    print(f"  Stopband freq    fs = {fs/1e9:.2f} GHz")
    print(f"  Passband ripple  Rp = {Rp} dB")
    print(f"  Stopband atten   Rs = {Rs} dB")
    print(f"  Impedance        R0 = {R0} Ohm")
    print("-" * 55)
    print(f"  Filter order     N  = {N}")
    print("-" * 55)
    print("  Normalized prototype g-values (fc=1 rad/s, R0=1 Ohm):")
    for i in range(N + 2):
        print(f"    g[{i}] = {g[i]:.6f}")
    print("-" * 55)
    print("  Element values (shunt-C / series-L topology):")
    print(f"  R_source = {R0} Ohm")
    for elem_type, k, val in elements:
        unit = 'F' if elem_type == 'C' else 'H'
        print(f"    {elem_type}{k} = {format_value(val, unit)}")
    print(f"  R_load   = {R0 * g[N+1]:.4f} Ohm")
    print("=" * 55)


def plot_response(N, Rp, fp, fs, Rs):
    """아날로그 Chebyshev Type I LPF 주파수 응답 플롯"""
    wc = 2.0 * np.pi * fp
    b, a = cheby1(N, Rp, wc, btype='low', analog=True)

    w = np.logspace(np.log10(wc * 0.05), np.log10(wc * 15), 4000)
    _, h = freqs(b, a, worN=w)
    freq_ghz = w / (2 * np.pi * 1e9)
    mag_db   = 20 * np.log10(np.abs(h))

    # fp, fs에서의 실제 감쇠값
    _, h_fp = freqs(b, a, worN=[2*np.pi*fp])
    _, h_fs = freqs(b, a, worN=[2*np.pi*fs])
    db_fp = 20 * np.log10(abs(h_fp[0]))
    db_fs = 20 * np.log10(abs(h_fs[0]))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.semilogx(freq_ghz, mag_db, linewidth=2, color='steelblue')

    # 기준선
    ax.axvline(fp/1e9, color='gray',   linestyle='--', linewidth=1.2, label=f'fp = {fp/1e9:.1f} GHz  ({db_fp:.2f} dB)')
    ax.axvline(fs/1e9, color='red',    linestyle='--', linewidth=1.2, label=f'fs = {fs/1e9:.1f} GHz  ({db_fs:.2f} dB)')
    ax.axhline(-Rp,    color='orange', linestyle=':',  linewidth=1.2, label=f'-Rp = -{Rp} dB')
    ax.axhline(-Rs,    color='blue',   linestyle=':',  linewidth=1.2, label=f'-Rs = -{Rs} dB')

    # fp, fs 교차점 마커
    ax.plot(fp/1e9, db_fp, 'o', color='gray', markersize=7, zorder=5)
    ax.plot(fs/1e9, db_fs, 'o', color='red',  markersize=7, zorder=5)

    # 교차점 수치 주석
    ax.annotate(f'{db_fp:.2f} dB', xy=(fp/1e9, db_fp),
                xytext=(fp/1e9 * 0.6, db_fp + 6),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')
    ax.annotate(f'{db_fs:.2f} dB', xy=(fs/1e9, db_fs),
                xytext=(fs/1e9 * 1.3, db_fs + 10),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    # x축 눈금: 주요 주파수 명시
    major_ticks = [0.1, 0.2, 0.5, 1.0, 2.0, 2.4, 5.0, 10.0, 15.0]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(v) for v in major_ticks], fontsize=8.5)
    ax.set_xlim(freq_ghz[0], freq_ghz[-1])

    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_title(f'Chebyshev Type I LPF  (N={N}, Rp={Rp} dB, fp={fp/1e9:.1f} GHz)')
    ax.set_ylim(-105, 5)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fp = 1e9    # 통과대역 주파수 [Hz]
    fs = 2.4e9  # 저지대역 주파수 [Hz]
    Rp = 0.2    # 통과대역 리플 [dB]
    Rs = 60.0   # 저지대역 감쇠 [dB]
    R0 = 50.0   # 특성 임피던스 [Ohm]

    N  = 7  # calc_order(fp, fs, Rp, Rs) -> 최소차수는 6, 50Ohm 정합을 위해 7로 고정
    g  = chebyshev_g_values(N, Rp)
    elements = scale_elements(g, N, fp, R0)
    print_results(N, g, elements, fp, fs, Rp, Rs, R0)
    plot_response(N, Rp, fp, fs, Rs)
