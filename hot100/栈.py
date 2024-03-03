# 20.有效的括号
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        d = {')':'(',']':'[','}':'{'}
        stack = []
        if not s:
            return True
        for c in s:
            if c in ['(','[','{']:
                stack.append(c)
            else:
                if stack and stack[-1] == d[c]:
                    stack.pop()
                else:
                    return False
        if not stack:
            return True
        return False

# 678. 有效的括号字符串
class Solution(object):
    def checkValidString(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack1, stack2 = [],[]
        for i in range(len(s)):
            if s[i] == '(':
                stack1.append(i)
            elif s[i] == '*':
                stack2.append(i)
            else:
                if stack1:
                    stack1.pop()
                elif stack2:
                    stack2.pop()
                else:
                    return False
        while stack1 and stack2:
            i1, i2 = stack1.pop(), stack2.pop()
            if i1 > i2:
                return False
        if stack1:
            return False
        return True

# 227. 基本计算器II
class Solution(object):
    def calculate(self, s):
        sign = '+'
        num = 0
        stack = []
        for i in range(len(s)):
            if s[i].isdigit() and s[i] != ' ':
                num = num * 10 + int(s[i])
            if s[i] in set('+-*/') or i == len(s) - 1:
                if sign == '+':
                    stack.append(num)
                elif sign == '-':
                    stack.append(-num)
                elif sign == '*':
                    stack.append(stack.pop() * num)
                else:
                    stack.append(int(stack.pop() * 1.0 / num ))
                num = 0
                sign = s[i]
        return sum(stack)

# 739. 每日温度
class Solution(object):
    def dailyTemperatures(self, temperatures):
        res = [0] * len(temperatures)
        s = []
        m = len(temperatures)
        for i in range(m - 1, -1, -1):
            while s and temperatures[s[-1]] <= temperatures[i]:
                s.pop()
            res[i] = s[-1] - i if s else 0
            s.append(i)
        return res

# 84. 柱状图中最大的矩形
class Solution(object):
    def largestRectangleArea(self, heights):
        n = len(heights)
        left, right = [0] * n, [0] * n

        mono_stack = list()
        for i in range(n):
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                mono_stack.pop()
            left[i] = mono_stack[-1] if mono_stack else -1
            mono_stack.append(i)

        mono_stack = list()
        for i in range(n - 1, -1, -1):
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                mono_stack.pop()
            right[i] = mono_stack[-1] if mono_stack else n
            mono_stack.append(i)

        ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n)) if n > 0 else 0
        return ans

# 402.移掉K位数字
class Solution(object):
    def removeKdigits(self, num, k):
        stack = []
        remain = len(num) - k
        for digit in num:
            while k and stack and stack[-1] > digit:
                stack.pop()
                k -= 1
            stack.append(digit)
        return ''.join(stack[:remain]).lstrip('0') or '0'
