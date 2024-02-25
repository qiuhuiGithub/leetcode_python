# 43. 字符串相乘
class Solution(object):
    def multiply(self, num1, num2):
        ans = ""
        for i in range(len(num2) - 1, -1, -1):
            res = int(num2[i])
            mul = res * int(num1) * (10 ** (len(num2)-i-1))
            ans = self.addStrings(ans, str(mul))
        return ans

    def addStrings(self, num1, num2):
        ans = ""
        carry = 0
        i, j = len(num1) - 1, len(num2) - 1
        while i >= 0 or j >= 0 or carry > 0:
            x = int(num1[i]) if i >= 0 else 0
            y = int(num2[j]) if j >= 0 else 0
            sum = x + y + carry
            carry = sum // 10
            sum = sum % 10
            ans += str(sum)
            i -= 1
            j -= 1
        return ans[::-1]

# 470. rand7() 实现rand10()
class Solution(object):
    def rand10(self):
        """
        :rtype: int
        """
        num = (rand7() - 1) * 7 + rand7()
        while num > 40:
            num = (rand7() - 1) * 7 + rand7()
        return 1 + num % 10
