from itertools import product

chars = ["c","a","t","d","o","g"]
"""idk if this should be a 'yield' exercise but this might also be pretty elegant(?)"""
for comb in product(chars, repeat=len(chars)): 
    print(''.join(comb))

    