class Flower:
    
    def __init__(self, name, numberofpetals, price): 
        self.name = name
        self.numberofpetals = numberofpetals
        self.price = price

    def get_name(self):
        return self.name
    
    def get_numberofpetals(self):
        return self.numberofpetals
    
    def get_price(self):
        return self.price

    def set_name(self, name):
        self.name = name
    
    def set_numberofpetals(self, numberofpetals):
        self.numberofpetals = numberofpetals

    def set_price(self, price):
        self.price = price

if __name__ == "__main__":

    Flower1 = Flower("Rose", 10, 5.0)
    Flower2 = Flower("Lily", 5, 3.0)
    Flower3 = Flower("Sunflower", 15, 7.0)

    print(Flower1.name, Flower1.numberofpetals, Flower1.price)
    print(Flower2.name, Flower2.numberofpetals, Flower2.price)
    print(Flower3.name, Flower3.numberofpetals, Flower3.price)

    print(Flower1.get_price())
    