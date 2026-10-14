"""Run ids and support draws (light: no Chameleon or torch imports; PLAN §16)."""


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
