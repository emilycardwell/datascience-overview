# If & Loops

## If Statement

```python
is_female = True
is_tall = False

if is_female:
		print("Gender is female")
else:
		print("Gender is not female")

if is_female or is_tall:
		print("You're female or tall")
else: 
		print("nope")

if is_female and is_tall:
		print("Congrats")
else:
		print("sorry")

if is_female and is_tall:
		print("tall lady")
elif is_female and not(is_tall):
		print("shawty")
elif not(is_female) and is_tall
		print("tall dude")
else:
		print("short dude")
```

### Advanced Calculator

```python
num1 = float(input("Enter first number: "))
op = input("Enter operator: ")
num2 = float(input("Enter second number: "))

if op == "+":
		print(num1 + num2)
elif op == "-":
		print(num1 - num2)
elif op == "/":
		print(num1 / num2)
elif op == "*":
		print(num1 * num2)
else:
		print("Invalid operator")
```

# Loops

### While

```python
i = 1
while i <= 10:
		print(i)
		i += 1
# adds one to i every loop while conditions are met

print("done")
```

### For

```python
range(start, stop, step)
```

```python
friends = ["Larissa", "Ana", "Amrit", "Komali", "Ellen"]
for friend in friends:
		print(friend)
#prints each element in array, string, etc.

for index in range(10):
		print(index)
#prints each number from 0 to 10, not uncluding 10

for index in range(len(friends)):
		 print(friends[index])
#prints each friend in ascending index number

for index in range(5):
		if index == 0:
				print("first")
		else:
				print("not first")
```

```python
# Python program to illustrate
# enumerate function in loops
l1 = ["eat", "sleep", "repeat"]
  
# printing the tuples in object directly
for ele in enumerate(l1):
    print (ele)
##(0, 'eat')
##(1, 'sleep')
##(2, 'repeat')
  
# changing index and printing separately
for count, ele in enumerate(l1, 100):
    print (count, ele)
##100 eat
##101 sleep
##102 repeat
  
# getting desired output from tuple
for count, ele in enumerate(l1):
    print(count)
    print(ele)
##0
##eat
##1
##sleep
##2
##repeat
```

### Nested For Loop

```python
number_grid = [
		[1, 2, 3],
		[4, 5, 6],
		[7, 8, 9],
		[0]
]

for row in number_grid:
		for col in row:
				print(col)
#1
#2
#3
#4
#5
#6
#7
#8
#9
#0
```

## Errors and Try/except

```python
EOFError # reading file
ZeroDivisionError # dividing by zero
TypeError # conversion issue
---
try:
		number = int(input("Enter a number: "))
		print(number)
except ValueError:
		print("Invalid Input")
---
try:
		number = int(input("Enter a number: "))
		answer = 10 / number
		print(answer)
except ZeroDivisionError as err:
		print(err)
# will print out the error
---
try:
		#code
except <Error1>:
		#handler
except <Error2>:
		#handler
else:
		# "code ran sucessfully"
finally:
		# do something in any case
---
try:
		raise Exception('An error occured!')
except Exception as error:
print(error)
---
class DogNotFoundException(Exception):
		pass

try:
		raise DogNotFoundException()
except DogNotFoundException:
		print('Dog not found!')
```