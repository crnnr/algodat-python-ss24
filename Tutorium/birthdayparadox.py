import random

def has_duplicates(birthdays):
    for i in range(len(birthdays)):
        for j in range(len(birthdays)):
            if i != j and birthdays[i] == birthdays[j]:
                return True
        
def birthdayparadox(n, num_simulation):
    duplicates = 0
    for i in range(num_simulation):
        birthdays = []
        for j in range(n):
              birthdays.append(random.randint(1, 365))
        if has_duplicates(birthdays):
            duplicates += 1
    return duplicates / num_simulation


if __name__ == "__main__":
    n = input("Enter the number of people: ")
    n = int(n)
    num_simulation = input("Enter the number of simulations: ")
    num_simulation = int(num_simulation)
    print(birthdayparadox(n, num_simulation))







    