def smalleroddsquares(n):
    numsum = 0
    while n > 0:
        n -= 1
        if n % 2 != 0:
            numsum += n**2
    print(numsum)

if __name__ == "__main__":
    smalleroddsquares(4)
            