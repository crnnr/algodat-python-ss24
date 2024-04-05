def is_even(k):

    """Return True if k is even, and False otherwise.
    
    ...01 if odd, 
    ...00 if even
    The number 1 in binary is 0001, and the number 2 in binary is 0010.

    0&1 = 0
    1&1 = 1

    
    """

    return k&1 == 0



if __name__ == "__main__":
   print(is_even(2))