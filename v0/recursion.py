# encoding=utf-8

# 50. Pow(x,n)
def myPow(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x
    if n == -1:
        return 1 / x
    half = myPow(x, n // 2)
    rest = myPow(x, n % 2)
    return half * half * rest

# 70.爬楼梯
def climbStairs(n):
    """
    :type n: int
    :rtype: int
    """
    # if n == 1:
    #     return 1
    # if n == 2:
    #     return 2
    # return climbStairs(n - 1) + climbStairs(n - 2)
    if n < 3:
        return n
    i1, i2 = 1, 2
    for i in range(3, n + 1):
        tmp = i1 + i2
        i1 = i2
        i2 = tmp
    return i2

def move(s, n):
    if not s:
        return ''
    n = n % len(s)
    m = len(s) - n
    s = s[n+1:] + s[0:m]
    return s