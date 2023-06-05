# 列表推导式
list1 = [x for x in range(1, 10)]
print('list1: {}'.format(list1))

list2 = [x + 1 for x in range(1, 10)]
print('list2: {}'.format(list2))

list3 = [x * x for x in range(1, 10)]
print('list3: {}'.format(list3))

list4 = [x for x in range(1, 10) if x % 2 == 0]
print('list4: {}'.format(list4))

list5 = [x * 10 for x in range(1, 10) if x % 2 == 0]
print('list5: {}'.format(list5))

list6 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
list7 = [x for y in list6 for x in y if x % 2 == 0]
print('list7: {}'.format(list7))

list8 = [x for x in 'python']
print('list8: {}'.format(list8))

list9 = [x + y for x in 'python' for y in '12']
print('list9: {}'.format(list9))

# 元组推导式
tuple1 = (x for x in range(1, 10))
print('tuple1: {}'.format(tuple1))

tuple2 = tuple(x for x in range(1, 10))
print('tuple2: {}'.format(tuple2))

# 字典推导式
dict1 = {x: x ** 2 for x in range(0, 9)}
print('dict1: {}'.format(dict1))

list10 = ['hello', 'python']
dict2 = {key: len(key) for key in list10}
print('dict2: {}'.format(dict2))

# 集合推导式
set1 = {x for x in 'abcde'}
print('set1: {}'.format(set1))
print('set1 type: {}'.format(type(set1)))
