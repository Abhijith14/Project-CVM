def sumofnumbers(n, a):
    sum = 0
    for i in range(n):
        sum += a[i]
    return sum

print(sumofnumbers(5, [1, 2, 3, 4, 5]))