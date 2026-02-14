"""
Headless bulk-shift computation engine for GeoAgent.

Wraps synthetic functions for non-GUI batch processing of
seismic-to-well tie bulk shifts (cross-correlation scan).

Usage:
    from geoagent.synthetic.bulk_shift import (
        compute_bulk_shift_scan,
        apply_and_save_bulk_shift,
        plot_synthetic_tie,
        audit_tdr_bulk_shifts,
    )
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from geoagent.synthetic.functions import (
    prepare_data_w_tdr,
    extract_seismic_in_range,
    create_reflectivity,
    create_synthetic_seismic_valid,
    calculate_correlation_with_shift,
    prepare_wavelet_for_synthetic,
)
from geoagent.utils.safe_print import safe_print


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_tdr_bulk_shifts(data_manager):
    """
    Inspect every well's TDR mapping and synthetic_settings to determine
    which wells have bulk-shifted TDRs.

    Returns:
        list[dict] with keys: well, checkshot_id, source_well, has_bs,
            bs_amount_ms, bs_cc, bs_analysis_done
    """
    tdr_mappings = data_manager.well_handler.get_all_tdr_mappings()
    syn_settings = data_manager.get_synthetic_settings_all() or {}
    well_settings = syn_settings.get('well_settings', {})

    rows = []
    for well, mapping in tdr_mappings.items():
        cs_id = mapping.get('checkshot_id', '')
        src = mapping.get('source_well', '')
        has_bs = '_BS_' in str(cs_id)

        # Parse BS amount from checkshot ID (e.g. "CS_BK-14_BS_-8")
        bs_amount = None
        if has_bs:
            try:
                bs_amount = float(cs_id.split('_BS_')[-1])
            except (ValueError, IndexError):
                pass

        # Pull from synthetic_settings
        ws = well_settings.get(well, {})
        bs_analysis_done = ws.get('bulk_shift_analysis_done', False)
        bs_cc = ws.get('bs_finder_correlation_coefficient', None)
        if bs_cc is None:
            synth = ws.get('synthetics', {}) or {}
            bs_cc = synth.get('bs_finder_correlation_coefficient', None)

        rows.append({
            'well': well,
            'checkshot_id': cs_id,
            'source_well': src,
            'has_bs': has_bs,
            'bs_amount_ms': bs_amount,
            'bs_cc': bs_cc,
            'bs_analysis_done': bs_analysis_done,
        })

    return rows


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def compute_bulk_shift_scan(
    data_manager,
    well_name,
    survey,
    attribute,
    shift_range=(-50, 50),
    shift_step=2,
    wavelet=None,
    extract_range=None,
):
    """
    Compute CC for each shift value — headless version of BulkShiftFinderDialog.

    Returns:
        dict with keys for CC scan results AND full synthetic tie data
        (log_df, well_tdr, impedance, reflectivity, wavelet, etc.)
        or None on failure.
    """
    try:
        return _compute_bulk_shift_scan_impl(
            data_manager, well_name, survey, attribute,
            shift_range, shift_step, wavelet, extract_range,
        )
    except Exception as e:
        safe_print(f"[BS] Unexpected error scanning {well_name}: {e}")
        return None


def _compute_bulk_shift_scan_impl(
    data_manager, well_name, survey, attribute,
    shift_range, shift_step, wavelet, extract_range,
):
    """Inner implementation — separated so the outer function can catch all errors."""
    # --- Get TDR & prepare data ----
    well_tdr = data_manager.well_handler.get_preferred_tdr(well_name)
    if well_tdr is None or (hasattr(well_tdr, 'empty') and well_tdr.empty):
        safe_print(f"[BS] No TDR for well {well_name}")
        return None

    prep = prepare_data_w_tdr(
        data_manager, well_tdr, well_name,
        use_upscaled=False,
        current_survey=survey,
        current_attribute=attribute,
        TRACE_INDEX=5,
    )
    if prep is None:
        safe_print(f"[BS] prepare_data_w_tdr failed for {well_name}")
        return None

    log_df = prep['log_df']
    log_times = prep['log_times']
    acoustic_impedance = prep['acoustic_impedance'].values if hasattr(prep['acoustic_impedance'], 'values') else prep['acoustic_impedance']
    seismic_trace = prep['seismic_trace']
    times = prep['times']
    current_sampling_interval = prep['current_sampling_interval']

    # Guard against ML-resolution sampling (0.25ms) leaking into synthetic-tie
    # computation.  Seismic is always 2ms; using sub-ms sampling corrupts the
    # wavelet convolution (wavelet stays 257 samples but covers 8x less time).
    seismic_dt = float(np.median(np.diff(times)))
    if current_sampling_interval < seismic_dt * 0.5:
        safe_print(f"[BS] Overriding sampling {current_sampling_interval} ms -> {seismic_dt} ms for {well_name}")
        current_sampling_interval = seismic_dt
        # Re-derive log_times at the correct sampling
        log_twt = log_df['TWT'].values if 'TWT' in log_df.columns else log_times
        old_twt = np.asarray(log_twt).flatten()
        t_start = float(np.min(old_twt))
        t_end = float(np.max(old_twt))
        new_times = np.arange(t_start, t_end + seismic_dt / 2, seismic_dt)
        # Interpolate AI onto the new time grid
        acoustic_impedance = np.interp(new_times, old_twt,
                                       np.asarray(acoustic_impedance).flatten())
        # Resample log_df onto the new time grid so plotting stays consistent
        new_log = {'TWT': new_times}
        for col in log_df.columns:
            if col == 'TWT':
                continue
            new_log[col] = np.interp(new_times, old_twt, log_df[col].values)
        log_df = pd.DataFrame(new_log)
        log_times = new_times

    # --- Resolve wavelet ---
    if wavelet is None:
        wavelet = _get_well_wavelet(data_manager, well_name)
    if wavelet is None:
        safe_print(f"[BS] No wavelet available for {well_name}")
        return None

    # --- Extract & build synthetic ---
    if extract_range is None:
        extract_range = (float(times[0]), float(times[-1]))

    (seismic_times_ext, seismic_trace_ext,
     well_times_ext, impedance_ext, reflectivity) = extract_seismic_in_range(
        seismic_trace, times, log_times, acoustic_impedance,
        bulk_shift=0, extract_range=extract_range,
        current_sampling_interval=current_sampling_interval,
    )

    synthetic_seismic = create_synthetic_seismic_valid(
        reflectivity, wavelet, seismic_times_ext,
        log_sampling_interval=current_sampling_interval,
    )

    # Align lengths (synthetic may be 1 sample shorter due to reflectivity diff)
    synthetic_times = well_times_ext[:len(synthetic_seismic)]
    synthetic_seismic = synthetic_seismic[:len(synthetic_times)]

    # --- Scan shifts ---
    shifts = np.arange(shift_range[0], shift_range[1] + shift_step, shift_step, dtype=float)
    correlations = np.zeros(len(shifts))

    for i, s in enumerate(shifts):
        correlations[i] = calculate_correlation_with_shift(
            seismic_trace_ext, seismic_times_ext,
            synthetic_seismic, synthetic_times,
            shift=s,
        )

    # --- Find best via local maxima (same logic as BulkShiftFinderDialog) ---
    local_max_idx = argrelextrema(correlations, np.greater)[0]

    if local_max_idx.size == 0:
        best_idx = int(np.argmax(correlations))
    else:
        best_local = local_max_idx[np.argmax(correlations[local_max_idx])]
        best_idx = int(best_local)

    best_shift = float(shifts[best_idx])
    best_cc = float(correlations[best_idx])

    local_maxima_shifts = shifts[local_max_idx].tolist() if local_max_idx.size > 0 else []
    local_maxima_ccs = correlations[local_max_idx].tolist() if local_max_idx.size > 0 else []

    # CC at zero shift
    zero_idx = int(np.argmin(np.abs(shifts)))
    cc_at_zero = float(correlations[zero_idx])

    # --- Resolve sonic/density column names for plotting ---
    sonic_col = data_manager.find_log_column('sonic_log', well_name)
    density_col = data_manager.find_log_column('density_log', well_name)

    return {
        # CC scan results
        'shifts': shifts,
        'correlations': correlations,
        'best_shift': best_shift,
        'best_cc': best_cc,
        'local_maxima_shifts': local_maxima_shifts,
        'local_maxima_ccs': local_maxima_ccs,
        'cc_at_zero': cc_at_zero,
        # Synthetic tie data (for plotting)
        'log_df': log_df,
        'log_times': log_times,
        'well_tdr': well_tdr,
        'synthetic_times': synthetic_times,
        'synthetic_seismic': synthetic_seismic,
        'seismic_times': seismic_times_ext,
        'seismic_trace': seismic_trace_ext,
        'well_times_ext': well_times_ext,
        'impedance_ext': impedance_ext,
        'reflectivity': reflectivity,
        'wavelet': wavelet,
        'current_sampling_interval': current_sampling_interval,
        'sonic_col': sonic_col,
        'density_col': density_col,
    }


# ---------------------------------------------------------------------------
# Apply & save
# ---------------------------------------------------------------------------

def apply_and_save_bulk_shift(data_manager, well_name, shift_ms):
    """
    Apply a constant bulk shift to the well's active TDR and save as a new
    checkshot record.  Mirrors SyntheticSeismicTieAppNew.save_bulk_shifted_tdr().

    Args:
        data_manager: DataManager instance
        well_name: target well
        shift_ms: shift in milliseconds (positive = deeper)

    Returns:
        new_checkshot_id (str) on success, None on failure
    """
    tdr_mapping = data_manager.well_handler.get_well_tdr(well_name)
    if not tdr_mapping or 'checkshot_id' not in tdr_mapping:
        safe_print(f"[BS] No TDR mapping for {well_name}")
        return None

    old_cs_id = tdr_mapping['checkshot_id']
    new_cs_id = f"{old_cs_id}_BS_{int(shift_ms)}"

    well_tdr = data_manager.well_handler.get_preferred_tdr(well_name)
    if well_tdr is None or well_tdr.empty:
        safe_print(f"[BS] Empty TDR for {well_name}")
        return None

    shifted_tdr = well_tdr.copy()
    shifted_tdr['TWT picked'] = shifted_tdr['TWT picked'] + shift_ms

    data_manager.well_handler.merge_checkshot_data(
        shifted_tdr, well_name, new_checkshot_id=new_cs_id,
    )

    safe_print(f"[BS] Saved shifted TDR for {well_name}: {new_cs_id}")
    return new_cs_id


# ---------------------------------------------------------------------------
# Plotting — Synthetic Tie Style (matches SyntheticSeismicTieAppNew layout)
# ---------------------------------------------------------------------------

def _extract_wavelet_amplitude(wavelet):
    """Extract the amplitude array from whatever wavelet format we have."""
    if isinstance(wavelet, tuple) and len(wavelet) == 2:
        data, meta = wavelet
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 2:
            return data[:, 1]  # column 1 = amplitude
        return np.asarray(data).flatten()
    elif isinstance(wavelet, dict):
        return np.asarray(wavelet.get('amplitude', wavelet.get('data', []))).flatten()
    else:
        return np.asarray(wavelet).flatten()


def _wavelet_time_axis(wavelet, sampling_interval, center_time):
    """Build a time axis for the wavelet, centred on center_time."""
    amp = _extract_wavelet_amplitude(wavelet)
    n = len(amp)
    t = np.arange(n) * sampling_interval - (n // 2) * sampling_interval
    return t + center_time


def plot_synthetic_tie(
    scan_result,
    well_name,
    output_path,
    formation_tops=None,
    figsize=(18, 10),
):
    """
    Generate an 8-panel synthetic-tie figure matching the SyntheticSeismicTieAppNew
    layout:  TDR | Sonic | Density | Impedance | Refl.Coeff | Seismic | Synthetic | Wavelet

    All panels share a common Y-axis (TWT, increasing downward).
    The best bulk shift is applied to the synthetic panel.
    Formation tops are drawn as horizontal lines across all panels.

    Args:
        scan_result: dict from compute_bulk_shift_scan()
        well_name: for title
        output_path: PNG path
        formation_tops: dict {name: twt_ms}
        figsize: figure size tuple
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    best_shift = scan_result['best_shift']
    best_cc = scan_result['best_cc']
    cc_at_zero = scan_result['cc_at_zero']

    # Unpack data
    well_tdr = scan_result['well_tdr']
    log_df = scan_result['log_df']
    log_times = scan_result['log_times']
    seismic_times = scan_result['seismic_times']
    seismic_trace = scan_result['seismic_trace']
    well_times_ext = scan_result['well_times_ext']
    impedance_ext = scan_result['impedance_ext']
    reflectivity = scan_result['reflectivity']
    synthetic_seismic = scan_result['synthetic_seismic']
    synthetic_times = scan_result['synthetic_times']
    wavelet = scan_result['wavelet']
    sampling = scan_result['current_sampling_interval']
    sonic_col = scan_result['sonic_col']
    density_col = scan_result['density_col']

    # Apply best shift to log times for display
    log_times_shifted = log_times + best_shift

    # --- Create figure with 8 panels ---
    fig, axes = plt.subplots(1, 8, figsize=figsize, sharey=True,
                             gridspec_kw={'width_ratios': [0.8, 1, 1, 1, 0.8, 1.2, 1.2, 0.8]})
    fig.subplots_adjust(wspace=0.08, top=0.88, bottom=0.06, left=0.04, right=0.98)

    title_str = f'{well_name}:  CC = {best_cc:.3f}  |  Best Shift = {best_shift:+.0f} ms  |  CC@0 = {cc_at_zero:.3f}'
    fig.suptitle(title_str, fontsize=12, fontweight='bold')

    fs = 8  # font size for labels/titles
    top_colors = ['#2E7D32', '#1565C0', '#C62828', '#6A1B9A', '#4E342E',
                  '#000000', '#E65100', '#37474F']

    # ---- Panel 0: TDR ----
    ax = axes[0]
    if 'Z' in well_tdr.columns and 'TWT picked' in well_tdr.columns:
        tdr_z = well_tdr['Z'].values
        tdr_twt = well_tdr['TWT picked'].values + best_shift
        ax.plot(-tdr_z, tdr_twt, color=top_colors[0], lw=1.0)
    ax.set_title('TDR', fontsize=fs, pad=10, color=top_colors[0])
    ax.set_xlabel('Depth (m)', fontsize=fs - 1)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=fs - 1)
    ax.invert_xaxis()

    # ---- Panel 1: Sonic ----
    ax = axes[1]
    if sonic_col and sonic_col in log_df.columns:
        ax.plot(log_df[sonic_col].values, log_times_shifted,
                color=top_colors[1], lw=0.7)
    ax.set_title('Sonic', fontsize=fs, pad=10, color=top_colors[1])
    ax.set_xlabel(f'{sonic_col or "DT"} (us/ft)', fontsize=fs - 1)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=fs - 1)

    # ---- Panel 2: Density ----
    ax = axes[2]
    if density_col and density_col in log_df.columns:
        ax.plot(log_df[density_col].values, log_times_shifted,
                color=top_colors[2], lw=0.7)
    ax.set_title('Density', fontsize=fs, pad=10, color=top_colors[2])
    ax.set_xlabel(f'{density_col or "RHOB"} (g/cc)', fontsize=fs - 1)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=fs - 1)

    # ---- Panel 3: Impedance ----
    ax = axes[3]
    imp = np.asarray(impedance_ext).flatten()
    wt = np.asarray(well_times_ext).flatten()[:len(imp)]
    ax.plot(imp, wt, color=top_colors[3], lw=0.7)
    ax.set_title('Impedance', fontsize=fs, pad=10, color=top_colors[3])
    ax.set_xlabel('AI', fontsize=fs - 1)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=fs - 1)

    # ---- Panel 4: Reflectivity ----
    ax = axes[4]
    refl = np.asarray(reflectivity).flatten()
    refl_times = np.asarray(well_times_ext).flatten()[:len(refl)]
    ax.plot([0, 0], [refl_times[0], refl_times[-1]], color='#90A4AE', lw=0.5)
    ax.hlines(refl_times, 0, refl, color=top_colors[4], lw=0.5)
    ax.set_title('Refl. Coeff', fontsize=fs, pad=10, color=top_colors[4])
    ax.set_xlabel('RC', fontsize=fs - 1)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=fs - 1)

    # ---- Panel 5: Seismic (actual) ----
    ax = axes[5]
    ax.plot(seismic_trace, seismic_times, color=top_colors[5], lw=0.7)
    ax.fill_betweenx(seismic_times, 0, seismic_trace,
                      where=(seismic_trace > 0), color='#1565C0', alpha=0.3)
    ax.fill_betweenx(seismic_times, 0, seismic_trace,
                      where=(seismic_trace < 0), color='#C62828', alpha=0.3)
    ax.set_title('Seismic', fontsize=fs, pad=10, color=top_colors[5])
    ax.set_xlabel('Amplitude', fontsize=fs - 1)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=fs - 1)

    # ---- Panel 6: Synthetic (at best shift) ----
    ax = axes[6]
    shifted_syn_times = synthetic_times + best_shift
    # Normalize synthetic to seismic amplitude range for overlay
    syn = np.asarray(synthetic_seismic).flatten()
    seis_for_overlay = np.interp(shifted_syn_times, seismic_times, seismic_trace)
    if np.std(syn) > 1e-12:
        syn_norm = syn / np.max(np.abs(syn)) * np.max(np.abs(seis_for_overlay))
    else:
        syn_norm = syn
    ax.plot(seis_for_overlay, shifted_syn_times, color='black', lw=0.6, alpha=0.5, label='Seismic')
    ax.plot(syn_norm, shifted_syn_times, color=top_colors[6], lw=0.8, label='Synthetic')
    ax.set_title('Synthetic', fontsize=fs, pad=10, color=top_colors[6])
    ax.set_xlabel('Amplitude', fontsize=fs - 1)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=fs - 1)
    ax.legend(fontsize=fs - 2, loc='lower right')

    # ---- Panel 7: Wavelet ----
    ax = axes[7]
    wv_amp = _extract_wavelet_amplitude(wavelet)
    mid_time = np.mean(seismic_times)
    wv_times = _wavelet_time_axis(wavelet, sampling, mid_time)
    ax.plot(wv_amp, wv_times, color=top_colors[7], lw=0.8)
    ax.set_title('Wavelet', fontsize=fs, pad=10, color=top_colors[7])
    ax.set_xlabel('Amplitude', fontsize=fs - 1)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=fs - 1)

    # ---- Common Y-axis setup ----
    axes[0].set_ylabel('TWT (ms)', fontsize=fs)
    ymin = float(seismic_times[0])
    ymax = float(seismic_times[-1])
    axes[0].set_ylim(ymax, ymin)  # invert: TWT increases downward

    # ---- Formation tops (horizontal lines across all panels) ----
    if formation_tops:
        top_palette = ['#2E7D32', '#1565C0', '#C62828', '#6A1B9A',
                       '#E65100', '#00695C', '#4E342E', '#AD1457']
        for idx, (name, twt) in enumerate(formation_tops.items()):
            if ymin <= twt <= ymax:
                clr = top_palette[idx % len(top_palette)]
                for ax in axes:
                    ax.axhline(twt, color=clr, ls='--', lw=0.7, alpha=0.7)
                # Label on the first and last panel
                axes[0].text(axes[0].get_xlim()[1], twt, f' {name}',
                             fontsize=fs - 2, color=clr, va='bottom', ha='left',
                             clip_on=True)

    # Grid on all panels
    for ax in axes:
        ax.grid(True, axis='y', ls='--', alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    safe_print(f"[BS] Saved figure: {output_path}")


# Keep old name as alias for backward compatibility
plot_bulk_shift_panel = plot_synthetic_tie


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _get_well_wavelet(data_manager, well_name):
    """
    Retrieve the wavelet used for this well's synthetic tie from
    synthetic_settings.  Falls back to imported project wavelets,
    then to a default 30 Hz Ricker.
    """
    syn_settings = data_manager.get_synthetic_settings_all() or {}
    well_settings = syn_settings.get('well_settings', {}).get(well_name, {})

    wv = well_settings.get('wavelet', None)
    if wv is not None:
        return wv

    # Fallback: check imported wavelets in seismic handler
    imported = data_manager.seismic_handler.loaded_data.get('wavelets', [])
    if imported:
        wv_entry = imported[0]  # use first imported wavelet
        if isinstance(wv_entry, dict) and 'data' in wv_entry:
            data = wv_entry['data']
            meta = wv_entry.get('metadata', {})
            safe_print(f"[BS] Using imported wavelet '{wv_entry.get('name', '?')}' for {well_name}")
            return (np.asarray(data), meta)

    # Last resort: 30 Hz Ricker
    safe_print(f"[BS] No saved wavelet for {well_name}, using default 30 Hz Ricker")
    from scipy.signal import ricker
    return ricker(128, 30.0 / (1000.0 / (128 * 2.0)))
