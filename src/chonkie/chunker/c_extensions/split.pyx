# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import cython
from cpython.list cimport PyList_New, PyList_Append
from cpython.unicode cimport PyUnicode_Replace

@cython.boundscheck(False)
@cython.wraparound(False)
def split_text(
    str text,
    delim=None,
    str include_delim="prev",
    int min_characters_per_segment=12,
    bint whitespace_mode=False,
    bint character_fallback=True
):
    """
    Unified text splitting function optimized with Cython.
    Uses the efficient approach from RecursiveChunker - keeps delimiters intact.
    
    Args:
        text: Input text to split
        delim: Delimiter(s) to split on. Can be str, List[str], or None
        include_delim: Where to include delimiter ("prev", "next", or None)
        min_characters_per_segment: Minimum characters per segment for merging
        whitespace_mode: If True and delim is None, split on whitespace
        character_fallback: If True and no delimiters, split by character chunks
    
    Returns:
        List of text segments
    """
    cdef:
        str sep = "✄"
        str modified_text = text
        str delimiter
        list delimiters = []
        list splits = []
        list segments = PyList_New(0)
        str segment
        int i, end_pos
        int text_len = len(text)
        bint found_delim = False
    
    # Handle None delim case
    if delim is None:
        if whitespace_mode:
            # Split on whitespace - for word-level splitting
            splits = text.split(" ")  # Split on spaces specifically, not all whitespace
            return _merge_short_segments_fast(splits, min_characters_per_segment)
        elif character_fallback and text_len > min_characters_per_segment:
            # Split into character chunks
            for i in range(0, text_len, min_characters_per_segment):
                end_pos = min(i + min_characters_per_segment, text_len)
                segment = text[i:end_pos]
                PyList_Append(segments, segment)
            return segments
        else:
            # Return original text as single segment
            PyList_Append(segments, text)
            return segments
    
    # Convert delim to list
    if isinstance(delim, str):
        delimiters = [delim]
    else:
        delimiters = list(delim)
    
    # Use the efficient RecursiveChunker approach: keep delimiters, add separator
    if include_delim == "prev":
        # Add separator AFTER each delimiter (delimiter + sep)
        for delimiter in delimiters:
            if delimiter in modified_text:
                found_delim = True
                modified_text = PyUnicode_Replace(modified_text, delimiter, delimiter + sep, -1)
    elif include_delim == "next":
        # Add separator BEFORE each delimiter (sep + delimiter)
        for delimiter in delimiters:
            if delimiter in modified_text:
                found_delim = True
                modified_text = PyUnicode_Replace(modified_text, delimiter, sep + delimiter, -1)
    else:
        # Replace delimiter with separator only
        for delimiter in delimiters:
            if delimiter in modified_text:
                found_delim = True
                modified_text = PyUnicode_Replace(modified_text, delimiter, sep, -1)
    
    # If no delimiters found, return original text as single segment
    if not found_delim:
        PyList_Append(segments, text)
        return segments
    
    # Split by separator and filter empty
    splits = [split for split in modified_text.split(sep) if split != ""]
    
    # Merge short segments using fast method
    return _merge_short_segments_fast(splits, min_characters_per_segment)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _merge_short_segments_fast(list splits, int min_characters):
    """
    Fast segment merging using the RecursiveChunker approach.
    Mirrors the exact logic from RecursiveChunker._split_text.
    """
    cdef:
        str current = ""
        list merged = PyList_New(0)
        str split
        int split_len, current_len
    
    if not splits:
        return splits
    
    for split in splits:
        split_len = len(split)
        current_len = len(current)
        
        if split_len < min_characters:
            # Add to current accumulator
            current += split
        elif current:
            # We have accumulated content, add the split and append
            current += split
            PyList_Append(merged, current)
            current = ""
        else:
            # Split is long enough and no accumulated content
            PyList_Append(merged, split)
        
        # Check if current accumulator is now long enough
        current_len = len(current)
        if current_len >= min_characters:
            PyList_Append(merged, current)
            current = ""
    
    # Add any remaining accumulated content
    if current:
        PyList_Append(merged, current)
    
    return merged