from collections import abc
from typing import Any

__all__ = [
    'compare',
    'partial_compare',
]

_LIST_TYPES = (abc.Sequence, )
_DICT_TYPES = (abc.Mapping, )
_STR_TYPES = (str, bytes, )


def compare(haystack: Any, subset: Any, strict: bool = False) -> bool:
    """
    Deep compare the data structure given as `haystack` with the data structure given as `subset`. This method
    descends `Sequence` and `Mapping` types. Use the `strict` parameter to make sure that the `haystack` and `subset`
    values are of the same type.

    :param haystack: The data structure used as a comparison reference
    :param subset: The data structure compared to the haystack
    :param strict: Strict data type matching
    :return: If the second parameter is equal to the first parameter
    """
    return _compare(haystack, subset, False, strict)


def partial_compare(haystack: Any, subset: Any, strict: bool = False) -> bool:
    """
    Deep compare the data structure given as `haystack` with the data structure given as `subset`. This method
    descends `Sequence` and `Mapping` types. This method checks if the data structure given as `subset` as actually a
    subset of the data structure given as `haystack`. Use the `strict` parameter to make sure that the `haystack`
    and `subset` values are of the same type.

    :param haystack: The data structure used as a comparison reference
    :param subset: The data structure compared to the haystack
    :param strict: Strict data type matching
    :return: If the second parameter is a subset of the first parameter
    """
    return _compare(haystack, subset, True, strict)


def _compare(haystack: Any, subset: Any, partial: bool, strict: bool) -> bool:
    """
    Deep compare the data structure given as `haystack` with the data structure given as `subset`. This method
    descends `Sequence` and `Mapping` types.

    :param haystack: The data structure used as a comparison reference
    :param subset: The data structure compared to the haystack
    :param partial: Subset data comparison
    :param strict: Strict data type matching
    :return: If the second parameter is equal to, or a subset of the first parameter
    """
    if strict:
        # check type if we are working on strict mode
        if not issubclass(type(haystack), type(subset)):
            return False

    # if we compare compare two dict types, we check each key of the haystack object to be equal to the
    # subset object. if we are working in partial mode, we ignore if some keys are missing on the subset object.
    # however we check if all keys of the subset object are existing on the haystack object.
    if isinstance(haystack, _DICT_TYPES) and isinstance(subset, _DICT_TYPES):
        return _compare_mapping(haystack, subset, partial, strict)

    # if we compare compare two list types, we check each value of the haystack object to be equal to the
    # subset object. if we are working in partial mode, we ignore if the subset list is shorter than the haystack list.
    elif isinstance(haystack, _LIST_TYPES) and not isinstance(haystack, _STR_TYPES) \
            and isinstance(subset, _LIST_TYPES) and not isinstance(subset, _STR_TYPES):
        return _compare_sequence(haystack, subset, partial, strict)

    # for any other type, we just compare the two values.
    else:
        return haystack == subset


def _compare_mapping(haystack: abc.Mapping, subset: abc.Mapping, partial: bool, strict: bool) -> bool:
    # check if all keys of the subset are also on the haystack object
    for key in subset.keys():
        if key not in haystack:
            return False

    # check and compare each value of the haystack to the corresponding value on the subset object
    for key in haystack.keys():
        # ignore missing keys on subset if we are in partial mode
        if partial and key not in subset:
            continue

        elif key not in subset:
            return False

        if not _compare(haystack[key], subset[key], partial, strict):
            return False

    return True


def _compare_sequence(haystack: abc.Sequence, subset: abc.Sequence, partial: bool, strict: bool) -> bool:
    haystack_slice = haystack[:]

    # if we do not partially compare the lists, we need to check if the lengths of the two lists to compare are
    # equal.
    if not partial and len(haystack) != len(subset):
        return False

    for subset_value in subset:
        haystack_slice_index = 0

        # find the index of the first value in the haystack slice list that equals to the current value of the
        # subset list. if the haystack slice list does not contain the value of the subset, the lists are not equal.
        for haystack_value in haystack_slice:
            haystack_slice_index += 1

            if _compare(haystack_value, subset_value, partial, strict):
                break
        else:
            return False

        # reduce the haystack slice list to the values that have not been compared yet.
        haystack_slice = haystack_slice[haystack_slice_index:]

    return True
