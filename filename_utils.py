import os
import re

from audio_normalize import slugify_filename, shorten_slug_words, TARGET_SAMPLE_RATE


def _get_short_entry(short_basename, name_maps):
    if not short_basename:
        return None
    return name_maps.short_entry_info.get(short_basename)


def _determine_export_sr(short_basename, fallback_sr, name_maps):
    entry = _get_short_entry(short_basename, name_maps)
    if entry and entry.get('preserve_48k'):
        return int(entry.get('original_sample_rate') or 48000)
    fallback = fallback_sr if fallback_sr is not None else TARGET_SAMPLE_RATE
    return int(fallback)


def _lookup_short_entry_for_path(path, name_maps):
    candidates = []
    if path:
        candidates.append(path)
        norm = os.path.normpath(path)
        if norm not in candidates:
            candidates.append(norm)
        forward = norm.replace('\\', '/')
        if forward not in candidates:
            candidates.append(forward)
        backward = norm.replace('/', '\\')
        if backward not in candidates:
            candidates.append(backward)
    for cand in candidates:
        entry = name_maps.file_short_info.get(cand)
        if entry:
            return entry
    return None


def _split_pass_suffix(stem):
    if not stem:
        return '', ''
    match = re.search(r'_pass\d+(?:_\d+)?', stem)
    if match:
        return stem[:match.start()], stem[match.start():]
    return stem, ''


def _resolve_short_prefix(prefix, name_maps, entry=None):
    if entry:
        short_val = entry.get('short')
        if short_val:
            return short_val
        original = entry.get('original')
        if original:
            mapped = name_maps.basename_short_map.get(original)
            if mapped:
                return mapped
    if prefix:
        if prefix in name_maps.short_to_orig_map:
            return prefix
        mapped = name_maps.basename_short_map.get(prefix)
        if mapped:
            return mapped
    slug_source = slugify_filename(prefix or 'item')
    return shorten_slug_words(slug_source)


def _resolve_short_output_stem(input_file, name_maps):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    prefix, suffix = _split_pass_suffix(base_name)
    entry = _lookup_short_entry_for_path(input_file, name_maps)
    short_prefix = _resolve_short_prefix(prefix, name_maps, entry)
    return f'{short_prefix}{suffix}'


def strip_pass_prefixes(name):
    # Remove any _passN or _passN_M sequences that may have been appended previously
    # e.g. '05 - Breathe Deeper_pass2_14_pass3_14' -> '05 - Breathe Deeper'
    return re.sub(r'(?:_pass\d+(?:_\d+)?)+', '', name)
