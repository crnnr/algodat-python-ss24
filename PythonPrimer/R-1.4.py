def returnlowsquares(n):
    nsum = 0
    while n > 0:
        n = n - 1
        nsum = nsum + (n*n)
    print(nsum)


if __name__ == "__main__":
    returnlowsquares(10)

