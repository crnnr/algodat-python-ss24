def minmax(data):
    """
    Returns the minimum and maximum value of a list.
    """
    min_num = max_num = data[0]

    for num in data[1:]:
        if num < min_num:
            min_num = num
        if num > max_num:
            max_num = num

    return min_num, max_num

if __name__ == "__main__":
    data = [1, -2, 3, 4, 5]
    print(minmax(data))