# Functions & Classes

```python
def say_hi():
		print("Hello User") 
# contents of a function **must** be indented

say_hi()
# you must call a funtion in order to execute it

def say_hi(name):
		print("Hello " + name)
say_hi("Emily")
#gives the funtion parameters

def lucky_numbers(numbers):
		for num in numbers:
				if num == 7:
						return True
		return False
#checks list [numbers] for lucky number 7

def options(arg1, arg2=1)
#assigning value makes it optional
```

## Return Statement

```python
def cube(num):
		return num*num*num
# return gives the value of the function back to the caller
# return ends the function, **so you can't put any more lines of code after it**

result = cube(4)
print(result)
```

### Comparisons

```python
def max_num(num1, num2, num3):
		if num1 >= num2 and num1 >= num3:
				return num1
		elif num2 >= num1 and num2 >= num3:
				return num2
		else:
				return num3
# uses comparisons to make true/false statements

print(max_num(3, 4, 5))

# not equal: != 
# if equal: ==
# >, <, >=, <=
```

### exponents

```python
print(2**3)
# ** creates an exponent (2^3 - 2 to the power of 3)

def raise_to_power(base_num, pow_num):
		result = 1
		for index in range(pow_num):
			result = result * base_num
		return result

print(raise_to_power(3, 8))
# 3 to the 8th power returned
```

# List Comprehension

```python
# my_new_list = [ expression for item in list ]

digits = [x for x in range(10)]
# create list
##[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# loop vs. comprehension
for x in range(10):
    squares.append(x**2)
print(squares)
#VS
squares = [x**2 for x in range(10)]
##[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

even_numbers = [ x for x in range(1,20) if x % 2 == 0]
# filtering a list
##[2, 4, 6, 8, 10, 12, 14, 16, 18]

nums = [x+y for x in [1,2,3] for y in [10,20,30]]
print(nums)
##[11, 21, 31, 12, 22, 32, 13, 23, 33]
```

```python
#strings
authors = ["Ernest Hemingway","Langston Hughes","Frank Herbert","Toni Morrison",
    "Emily Dickson","Stephen King"]
letters = [name[0] for name in authors]
print(letters)
##['E', 'L', 'F', 'T', 'E', 'S']
```

```python
numbers_power_2 = []
for n in number:
		numbers_power_2.append(n**2)

numbers_power_2 = list(map(lambda n : n**2, numbers))
# maps a list from function taking each n -> n squared from numbers
```

# Lambda

```python
add_one = lambda x: x + 1
# equivalent to:
def add_one(x):
    return x + 1

sorted_nested_list = sorted(list_name, key = lambda x: x[i])
nested_list.sort(key = lambda x : x[1])
# i is the value you want it to be sorted at (1 would be second value)
```

# Recursion

- a function calling itself
- factorial: number = n → n  * (n-1) * (n-2) … → 1

```python
3! = 6 
# 3 * 2 * 1

def factorial(n):
		if n == 1: return 1
		return n * factorial(n - 1)
## factorial(3) = 6

# if you put factorial(n), you get RecursionError 
# (infinite recursion)

def count_down(t):
    if t == 0: return "Go!"
    print(t)
    return count_down(t-1)
print(count_down(10))
# 10
# 9
#...
# "Go!"
```

```python
def digital_root(n):
    if n == 0:
        return 0
    elif n < 10:
        return n
    else:
        numbers = [int(x) for x in str(n)]
        s = 0
        for i in numbers:
            s += i
        return digital_root(s)
# summing digits in an int until the sum is one digit
```

# Nested Functions

- it's always best to hide functionality that's local to a function, and is not useful elsewhere.
- make use of closures

```python
def talk(phrase):
		def say(word):
				print(word)
		
		words = phrase.split()
		for word in words:
				say(word)

talk("I am going to the store")
## I
## am
## ...

def count():
		count = 0

		def increment():
			nonlocal count
			# allows use outside of inner function
			count = count + 1
			print(count)

		increment()

count()
```

# Closures

- if you return nested function from outer function, the nested function has access to the variables within

# Decorators

- change/alter a function

```python
import datetime

def logtime(func):
		def inner():
				func()
				now = datetime.datetime.now()
				print(now)
		return inner

@logtime
def hello():
		print("hello")
		return

hello()
## hello
## 2022-09-21 18:38:44.966593
```

# Classes

- **OOP** - object oriented programming
    - data/state of behavior
    - **state**: stored in instance attributes (self.mammal)
    - **behavior**: defined by instance methods, functions (def is_mammal(self):) in class definition

```python
# Generalized interface:

# define method in different classes
class Dog:
		def eat():
				print("Eating dog food")

class Cat:
		def eat():
				print("Eating cat food")

# generate objects and call method regardless of class
animal1 = Dog()
anumal2 = Cat()

animal1.eat()
anumal2.eat()
```

- **instantiated**: the creation of a real instance or particular realization of an abstraction or template

```python
class Animal:
		pass
<Animal.py>

dog = Animal()
# creates instance of the class Animal
type(dog)
# __main__.Animal

Animal()
# instantintes the class, then calls the function '__init__'
class Animal:
		def __init__(self):
		# contructor method
				print("You have just instantiated the class!"

dog = Animal()
## You have just instantiated the class!
```

### Instance Attributes

```python
class Animal:
		def __init__(self):
				self.mammal = False
my_animal = Animal()
my_animal.mammal
## False
```

### Behavior

- **self**: instance on which the method has been called

```python
class Animal:
		def __init__(self):
				self.mammal = False
		def is_mammal(self):
				return self.mammal
my_animal = Animal()
my_animal.is_mammal()
## False
# don't forget () at end of mathod/funct
# BUT, when accessing value of attribute, no ()

class Animal:
		def __init_(self):
				self.mammal = False
		def is_mammal(self):
				return self.mammal
		def yes_mammal(self):
				self.mammal = True
my_animal = Animal()
my_animal.mammal
## False
# Calls attribute
my_animal.is_mammal()
## False
# Calls method
my_animal.yes_mammal()
# Calls method and initiates it, but doesn't print
my_animal.mammal
## True
# calls attribute
```

```python
class House:
		def __init__(self, name, main_tenant):
				self.name = name
				self.main_tenant = main_tenant

		def house_details(self):
				return f"{self.name} must pay rent to {self.main_tenant_name()}"

		def main_tenant_name(self):
				return self.main_tenant.capitalize()

ebersstraße43 = House("Emily", "angelika")
ebersstraße43.house_details()
## "Emily must pay rent to Angelika"
```

### Methods

```python
__init__()
# lets class initialize object's attributes

__gt__() (>)
# greater than

__eq__() (==)
# equal to

__lt__()
# lower than (<)

__le__()
# lower or equal (<=)

__ge__()
# greater or equal (>=)

__ne__()
# not equal (!=)

__add__() respond to the + operator
__sub__() respond to the – operator
__mul__() respond to the * operator
__truediv__() respond to the / operator
# float division
__floordiv__() respond to the // operator
# floor division
__mod__() respond to the % operator
__pow__() respond to the ** operator
```

### Bitwise Operators

```python
# shift bits by pushing zeros
__rshift__() respond to the >> operator
__lshift__() respond to the << operator
# sets each bit to 1 if both bits are 1
__and__() respond to the & operator
# sets each bit to 1 if either bit is 1
__or__() respond to the | operator
# sets each bit to 1 if only one of two bits is 1
__xor__() respond to the ^ operator
```

# Operator Overloading

```python
class Dog:
		def __init__(self, name, age):
				self.name = name
				self.age = age
		def __gt__(self, other):
				return True if self.age > other.age else False

roger = Dog("Roger", 8)
syd = Dog("Syd", 7)

print(roger > syd)
```

### Chaining

```python
cat = Cat()
cat.age_10_years().gain_weight().turn_grey()
```

# Inheritance

```python
class Animals:
		def __init__(self, name, blooded, livebirth, bipedal):
				self.name = name
				self.blooded = blooded
        self.livebirth = livebirth
        self.bipedal = bipedal

		def is_mammal(self):
				if blooded == "warm" and livebirth = True:
						return True
				else:
						return False 

class Mammals(Animals):
		pass
```

### Specific Behavior

```python
class Mammals(Animals):
		def __init__(self, name, blooded, livebirth, bipedal):
				super().__init__(same, name, blooded, livebirth, bipedal)
				self.tail = None

		def has_tail(self):
				return self.tail is not None

sheep = Mammals("Sheep", "warm", True, False)
sheep.has_tail()
## False
sheep.tail = "Yes"
sheep.has__tail()
## True
```

### Change arguments

```python
class Mammals(Animals):
		def __init__(self, name, bipedal, color):
				super().__init__(name, bipedal)
				self.color = color
```

### Change inherited method

```python
class Mammal(Animals):
		def is_mammal(self):
				return super().is_mammal() = True

sheep = Mammal("sheep", "warm", True, False)
sheep.is_mammal()
## True
```

### Class Functions

- when function does isn’t relevant to a single instance

```python
class Mammals:
		def __init__(self, name, tail):
				self.name = name			
				self.tail = tail

		def has_tail(self):
				return self.tail

pig = Mammal("pig", True)
pig.has_tail()
## True
# has_tail() is instance function

Mammal.has_tail()
## TypeError, class not an instance

Mammal.categories

class Mammals(Animal):

		@classmethod
		def categories(self:
				return ["Marine", "Land", "Flying"]

print(Mammals.categories())
##["Marine", "Land", "Flying"]

class House:
		@classmethod
		def price_per_sqm(self, city):
				if city is "Paris":
						return 9000
				elif city is "Brussels":
						return 3000
				else:
						return f"No data for {city}
```