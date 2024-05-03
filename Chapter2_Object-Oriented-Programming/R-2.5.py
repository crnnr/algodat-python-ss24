class CreditCard:
    
    balance = 0
    limit = 2000

    def __init__(self,balance,limit):
        self.balance = balance
        self.limit = limit


    def make_payment(self, amount):
        
        if amount is int:
            if amount > 0:
                self.balance =- amount
            else:
                raise ValueError('Work with Positive Numbers Only')
        else:
            raise TypeError('Work with Numbers Only')
        
        
        

    def get_balance(self):
        print(self.balance)

if __name__ == "__main__":

    Card1 = CreditCard(700,2000)
    Card1.get_balance()
    Card1.make_payment(-200)
    Card1.get_balance()

   