"""Recompute signed contrast and half-height widths from the saved paired patches on CPU."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_contrast(image, core, ring):
    return image[core].mean() - np.median(image[ring])


def direct_width(change, contrast_change, core, ring):
    footprint = core | ring
    relative = change - np.median(change[ring])
    highest = np.max(relative[core])
    gate = max(0.1, 3 * 1.4826 * np.median(np.abs(relative[ring])))
    if not np.any(change[footprint]):
        return None, 'identical_local_outputs'
    if contrast_change <= 0:
        return None, 'nonpositive_contrast_increment'
    if highest <= gate:
        return None, 'peak_below_local_noise_gate'
    if highest < np.max(relative[ring]):
        return None, 'local_peak_outside_injection_core'
    if highest / 2 <= gate:
        return None, 'half_height_below_local_noise_gate'
    allowed = set(zip(*np.nonzero(footprint)))
    selected = set(zip(*np.nonzero((relative >= highest / 2) & footprint)))
    seed = tuple(np.argwhere(core & (relative == highest))[0])
    stack = [seed]
    seen = {seed}
    touches_edge = False
    while stack:
        y, x = stack.pop()
        for yy in range(y - 1, y + 2):
            for xx in range(x - 1, x + 2):
                point = (yy, xx)
                if point not in allowed:
                    touches_edge = True
                elif point in selected and point not in seen:
                    seen.add(point)
                    stack.append(point)
    if len(seen) == 1:
        return None, 'single_pixel_unresolved'
    if touches_edge:
        return None, 'half_height_contour_touches_window_edge'
    return np.sqrt(4 * len(seen) / np.pi), 'measured'


def audit():
    index = json.loads((HERE / 'pairs.json').read_text())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    summary = json.loads((HERE / 'summary.json').read_text())
    with (HERE / 'per_pair_stage.csv').open() as stream:
        rows = {(r['sample_id'], r['route']): r for r in csv.DictReader(stream)}
    stages = index['stage_names']
    sources = [r['sample_id'] for r in protocol['pairs'] if r['pair_valid']]
    assert len(set(sources)) == len(sources) == 213 and len(stages) == len(set(stages)) == 5
    assert [r['sample_id'] for r in index['samples']] == sources
    assert set(rows) == {(source, stage) for source in sources for stage in stages}
    assert digest(HERE / 'per_pair_stage.csv') == summary['rows_sha256']
    assert digest(HERE / 'pairs.json') == summary['pair_archive_index_sha256']
    assert digest(HERE / 'protocol.json') == summary['protocol_sha256']
    counts, unchanged_codes = Counter(), 0
    for s in index['samples']:
        path = HERE / s['patch_file']
        assert digest(path) == s['patch_sha256']
        if sum(s['vq_changed_by_scale']) == 0:
            unchanged_codes += 1
            for stage in ('q_base_e_only', 'q_global_cac_fixed_h_minus'):
                frame = s['full_frame'][stage]
                assert frame['pixels'] == frame['equal_pixels'] and frame['max_abs_change_dn'] == 0
        with np.load(path, allow_pickle=False) as patch:
            core, ring = patch['core'], patch['ring']
            assert core.any() and ring.any() and not np.any(core & ring)
            for stage in stages:
                row = rows[s['sample_id'], stage]
                a = patch[stage + '_minus'].astype(np.float64) * 255
                b = patch[stage + '_plus'].astype(np.float64) * 255
                assert a.shape == b.shape == core.shape == ring.shape
                c = direct_contrast(b, core, ring) - direct_contrast(a, core, ring)
                np.testing.assert_allclose(c, float(row['incremental_contrast_dn']), atol=1e-12, rtol=0)
                if stage == 'reference_pair':
                    denominator = c
                    assert denominator > 0
                ratio = c / denominator
                np.testing.assert_allclose(ratio, float(row['contrast_retention_ratio']), atol=1e-12, rtol=0)
                width, reason = direct_width(b - a, c, core, ring)
                assert reason == row['width_reason']
                if width is None:
                    assert row['width_px'] == ''
                else:
                    np.testing.assert_allclose(width, float(row['width_px']), atol=1e-12, rtol=0)
                    counts[stage, 'width'] += 1
                counts[stage, 'retained'] += int(ratio >= 0.5)
                counts[stage, 'negative'] += int(ratio < 0)
    out = {}
    for stage in stages:
        actual = summary['by_nominal_peak_dn']['all'][stage]
        out[stage] = {k: counts[stage, k] for k in ('retained', 'negative', 'width')}
        assert list(out[stage].values()) == [actual['n_retained_half'], actual['n_negative_gain'], actual['width_px']['n']]
    assert unchanged_codes == summary['n_unchanged_vq_codes']
    result = dict(n_sources=len(sources), n_rows=len(rows), archive_hashes_checked=len(sources),
                  n_code_identical_with_verified_identical_full_frame_base_and_fixed_cac=unchanged_codes,
                  independent_contrast_and_width_recomputation='pass',
                  summary_sha256=digest(HERE / 'summary.json'), counts=out)
    (HERE / 'independent_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    audit()
