"""Run ids and support draws (light: no Chameleon or torch imports; PLAN §16)."""
import os


def run_name(dataset, shot, seed, sel='stratified', res='r768', tag='', support_seed=None):
    """run id: <dataset>_<shot>shot_s<seed>_<sel>[_<res>][_sup<k>][_<tag>]; _sup<k> only when the support draw
    (supports/<dataset>_<sel>_s<k>.txt) differs from the training seed (PLAN §16: seeds and draws vary separately)"""
    ss = seed if support_seed is None or support_seed < 0 else support_seed
    return f'{dataset}_{shot}shot_s{seed}_{sel}' + ('' if res == 'r768' else f'_{res}') \
        + ('' if ss == seed else f'_sup{ss}') + (f'_{tag}' if tag else '')


def support_seed_of(config):
    """support draw of a checkpoint's config (older checkpoints: the training seed)"""
    ss = getattr(config, 'rr_support_seed', None)
    return config.seed if ss is None else ss


def check_supports(dataset, ids, splits_dir):
    """stop when a training photo is also a tuning or confirmation photo (val30, confirm100): RescueNet's val30 comes
    from the same val pool as its supports and was cut outside draw 0 only, so its draws 1-2 of 5 Oct overlap it
    (PLAN §16.7; draws 3-4 are drawn without it). FloodNet's supports are train photos, its val30 val photos."""
    for name in (f'{dataset}_val_tune30', f'{dataset}_val_confirm100'):
        f = os.path.join(splits_dir, f'{name}.txt')
        if os.path.exists(f):
            bad = sorted(set(ids) & set(open(f).read().split()))
            if bad:
                raise SystemExit(f'{len(bad)} of the {len(ids)} support photos are in {name} ({", ".join(bad[:5])}): '
                                 'a run must not train on the photos that tune or confirm it; use another draw')
