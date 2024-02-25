# 301. 删除无效括号
class Solution(object):
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """

        def isValid(s):
            count = 0
            for c in s:
                if c == '(':
                    count += 1
                elif c == ')':
                    count -= 1
                    if count < 0:
                        return False
            return count == 0

        ans = []
        curSet = set([s])
        while True:
            for cur in curSet:
                if isValid(cur):
                    ans.append(cur)
            if len(ans) > 0:
                return ans
            tmp = set()
            for cur in curSet:
                for i in range(len(cur)):
                    if i > 0 and cur[i] == cur[i - 1]:
                        continue
                    if cur[i] in ['(', ')']:
                        tmp.add(cur[:i] + cur[i + 1:])
            curSet = tmp
        return ans