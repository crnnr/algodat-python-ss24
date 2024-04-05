def smallerthantwo(x):
    c = 0

    try:
        if x < 2:
            raise ValueError
    except ValueError:
        print("The number must be greater than 2.")
        return None
    if type(x) != int:
        print("The number must be an integer.")
        return None
    while x > 2:
        x = x / 2
        c += 1
    return c


if __name__ == "__main__":
    x = input("Enter a number: ")
    x = int(x)
    print(smallerthantwo(x))