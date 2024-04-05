
class Basic:
    def __init__(self):
        pass

    def hello(self):
        print('Hello, World!')
        
class Advanced(Basic):
    def __init__(self):
        super().__init__()

    def hello(self):
        print('Hello, World! from Advanced')

def main():
    basic = Basic()
    basic.hello()

    advanced = Advanced()
    advanced.hello()

if __name__ == '__main__':
    main()



''' def main():
    pass
    print('Hello, World!')

if __name__ == '__main__':
    main()''' 