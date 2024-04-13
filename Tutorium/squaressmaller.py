def sumsmallersquares(n=5):
    s = 0
    while n > 0:
        n = n - 1
        s += n**2
    return s


if __name__ == "__main__":
    inp = input("Enter a number: ")
    inp = int(inp)
    print(sumsmallersquares(inp))