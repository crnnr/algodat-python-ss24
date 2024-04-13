def minmax(data):
    for number in data:
        for i in data:
            if i < number:
                number = i
    for maxnum in data:
        for i in data:
            if i > maxnum:
                maxnum = i
    print("("+str(number)+","+str(maxnum)+")")


if __name__ == "__main__":
    data =  [5,-1,3,10,4,6]
    minmax(data)
