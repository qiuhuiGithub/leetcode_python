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