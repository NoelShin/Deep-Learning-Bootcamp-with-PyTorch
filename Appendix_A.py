class Calculator:
    def __init__(self):  # __init__ function is called "constructor" as every time you make an instance of Calculator,
                         # codes written in the constructor will be implemented.
        self.value = 0

    def add(self, n):
        self.value += n
        print("After addition:", self.value)

    def subtract(self, n):
        self.value -= n
        print("After subtraction:", self.value)

    def multiply(self, n):
        self.value *= n
        print("After multiplication:", self.value)

    def divide(self, n):
        if n == 0:
            print("You can't divide with 0.")
            return

        else:
            self.value /= n
        print("After division", self.value)

    def reset(self):
        self.value = 0
        print("reset value to", self.value)


print("Calculator a...")
a = Calculator()  # Make an instance of Calculator class. You need parenthesis after class name to do this.

a.add(5)  # Call add function defined in Calculator class.
a.divide(0)
a.divide(5)

print()
print("Calculator b...")
b = Calculator()  # Make another instance of Calculator class. This does not share value with calculator a.

b.add(1)
b.multiply(100)
b.reset()
