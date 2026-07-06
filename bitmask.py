from model_data import ITERATIVE_MODEL_KEYS


def _flag_order():
    """Ordered flag names (most-significant -> least-significant bit).

    The amplify bit must stay the most significant bit: compute_effective_mask
    clears it for non-final passes so intermediate pass folders can be reused
    regardless of the amplify setting.
    """
    order = ['amplify']
    order += [f'2x_{k}' for k in ITERATIVE_MODEL_KEYS]
    order += [f'mid_{k}' for k in ITERATIVE_MODEL_KEYS]
    order += ['post_bs_resurrect', 'post_scnet', 'side_restore']
    order += [f'use_{k}' for k in ITERATIVE_MODEL_KEYS]
    return order


MASK_FLAG_ORDER = _flag_order()
MASK_BIT_COUNT = len(MASK_FLAG_ORDER)
AMPLIFY_BIT_MASK = 1 << (MASK_BIT_COUNT - 1)
LOWER_BITS_MASK = AMPLIFY_BIT_MASK - 1


def build_mask(cfg):
    """Encode the run configuration into the folder-naming bitmask."""
    values = {
        'amplify': cfg.amplify_masked_details,
        'post_bs_resurrect': cfg.post_separate_bs_resurrect,
        'post_scnet': cfg.post_separate_scnet,
        'side_restore': cfg.restore_side_iterative,
    }
    for k in ITERATIVE_MODEL_KEYS:
        values[f'use_{k}'] = cfg.model_enabled(k)
        values[f'2x_{k}'] = cfg.slowdown_enabled(k)
        values[f'mid_{k}'] = cfg.mid_enabled(k)

    mask = 0
    for name in MASK_FLAG_ORDER:
        mask = (mask << 1) | int(bool(values.get(name, False)))
    return mask


def compute_effective_mask(mask, iteration_target, iterations_amount):
    """Return the effective mask for a given iteration.

    - For pass1 always return 0 (universal first-pass reuse).
    - For intermediate passes (2 .. iterations_amount-1) clear the amplify
        bit so amplify_masked_details does not change those pass folders.
    - For the final pass (iteration_target == iterations_amount) return the
        full mask (including amplify bit).
    """
    if iteration_target == 1:
        return 0
    # If this is not the final pass, clear the amplify bit so it only affects
    # the finisher (final) pass.
    if iteration_target < iterations_amount:
        return mask & LOWER_BITS_MASK
    return mask


def model_active_for_iteration(model_key, iteration_target, models_iterative_stage, iterations_amount):
    """Return True when an iterative-stage model should run this pass."""
    info = models_iterative_stage.get(model_key)
    if not info:
        return True
    limit = info.get('use_up_to_pass', iterations_amount)
    try:
        limit = int(limit)
    except Exception:
        limit = iterations_amount
    if limit < 1:
        return False
    return iteration_target <= limit
