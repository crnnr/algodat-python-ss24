def comprehensionrange(start,stop,step):
    [print(2**x) for x in range(start,stop,step)]

if __name__ == "__main__":
    comprehensionrange(0, 9, 1)