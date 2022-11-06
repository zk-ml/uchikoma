
def product(arr_like):
    """ calculate production for input array.

        if array is empty, return 1.
    """
    total = 1
    for s in arr_like:
        total *= s
    return total
