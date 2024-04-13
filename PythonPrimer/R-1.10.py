def makething(start,stop,step):
    ranging = range(start,stop,step)
    for n in ranging:
        print(n) 

if __name__ == "__main__":
    makething(8,-9,-2)