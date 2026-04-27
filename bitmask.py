from model_data import MASK_BIT_COUNT, AMPLIFY_BIT_MASK, LOWER_BITS_MASK


def mask_from_flags(amplify, d2_m1, d2_m2, d2_m3, d2_m4, post_separate_bs_resurrect, post_separate_scnet, side_restore, m1, m2, m3, m4):
    """Encode flags into a bitmask.

    Bit order (most-significant -> least-significant):
        amplify_masked_details,
        2x_m1, 2x_m2, 2x_m3, 2x_m4,
        post_separate_bs_resurrect, post_separate_scnet,
        side_restore,
        m1, m2, m3, m4

    Extends original scheme with four 2x slowdown bits and two post-separate flags.
    """
    bits = [
        int(amplify),
        int(d2_m1),
        int(d2_m2),
        int(d2_m3),
        int(d2_m4),
        int(post_separate_bs_resurrect),
        int(post_separate_scnet),
        int(side_restore),
        int(m1),
        int(m2),
        int(m3),
        int(m4),
    ]
    s = ''.join(str(b) for b in bits)
    return int(s, 2)


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
