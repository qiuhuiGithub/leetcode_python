# encoding=utf-8
# 20.有效的括号
def isValid(s) -> bool:
    if not s:
        return True
    stack = []
    dict = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in dict.values():
            stack.append(char)
        if char in dict.keys():
            if stack and stack[-1] == dict[char]:
                stack.pop()
            else:
                return False
    return stack == []

# 32.最长有效括号
def longestValidParentheses(s):
    longest = 0
    if not s:
        return longest
    stack = [-1]
    for i in range(len(s)):
        if s[i] == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                longest = max(longest, i - stack[-1])
    return longest