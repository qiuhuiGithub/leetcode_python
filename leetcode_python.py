# encoding=utf-8

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


# 1. two sum
def twoSum(nums, target):
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]


# print(twoSum([3, 2, 4], 6))

# 2. add two numbers
def addTwoNumbers(l1, l2):
    head = ListNode(0)
    result = head
    add = 0
    while l1 or l2:
        x = l1.val if l1 else 0
        y = l2.val if l2 else 0
        sum = x + y + add
        add = sum // 10
        result.next = ListNode(sum % 10)
        result = result.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    if add != 0:
        result.next = ListNode(1)
    return head.next


# l1 = ListNode(5)
# l1.next = ListNode(4)
# l1.next.next = ListNode(3)
# l2 = ListNode(5)
# l2.next = ListNode(6)
# l2.next.next = ListNode(4)
# result = addTwoNumbers(l1, l2)
# while result:
#     print(result.val)
#     result = result.next

# 3. 最长重复子串
def lengthOfLongestSubstring(s):
    ans = 0
    map = {}
    start = 0
    for j in range(len(s)):
        if s[j] in map.keys():
            start = max(map[s[j]] + 1, start)
        ans = max(ans, j - start + 1)
        map[s[j]] = j
    return ans


# print(lengthOfLongestSubstring('abcdb'))

# 4. 找两个数组的中位数
def findMedianSortedArrays(nums1, nums2):
    nums3 = []
    i = j = 0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] < nums2[j]:
            nums3.append(nums1[i])
            i += 1
        else:
            nums3.append(nums2[j])
            j += 1
    if i == len(nums1):
        nums3.extend(nums2[j:])
    else:
        nums3.extend(nums1[i:])
    if len(nums3) % 2 == 0:
        return (nums3[len(nums3) // 2 - 1] + nums3[len(nums3) // 2]) / 2
    else:
        return nums3[len(nums3) // 2]


# print(findMedianSortedArrays([1, 2], [3, 4]))

# 5. 最长回文子串
def longestPalindrome(s):
    if not s:
        return s
    dp = [[0] * len(s) for _ in range(len(s))]
    left, right = 0, 0
    for i in range(len(s) - 2, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, len(s)):
            dp[i][j] = 1 if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]) else 0
            if dp[i][j] and right - left < j - i:
                left = i
                right = j
    return s[left:right + 1]


# print(longestPalindrome('abcba'))


# 6. z字形变换
def convert(s, numRows):
    if numRows == 1:
        return s
    res = [''] * min(numRows, len(s))
    cur_row = 0
    go_down = -1
    for c in s:
        res[cur_row] += c
        if cur_row == 0 or cur_row == numRows - 1:
            go_down = -go_down
        cur_row += go_down
    return ''.join(res)


# print(convert('ABC', 4))

# 7. 整数反转
def reverse(x):
    if x == 0:
        return 0
    if x > 0:
        s = str(x)[::-1]
        res = int(s[1:]) if s[0] == 0 else int(s)
    else:
        s = str(x)[1:][::-1]
        res = -1 * int(s[1:]) if s[0] == 0 else -1 * int(s)
    if res > 2 ** 31 - 1 or res < -2 ** 31:
        return 0
    return res


# print(reverse(1534236469))

# 8.字符串转数字
def myAtoi(str):
    str = str.strip()
    if not str:
        return 0
    if str[0] not in ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        return 0
    minus = 1
    if str[0] == '-':
        minus = -1
        str = str[1:]
    elif str[0] == '+':
        str = str[1:]
    for i in range(len(str)):
        if str[i] != '0':
            str = str[i:]
            break
    if not str:
        return 0
    res = ''
    for char in str:
        if char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            res += char
        else:
            break
    if not res:
        return 0
    res = minus * int(res)
    if res > 2 ** 31 - 1:
        return 2 ** 31 - 1
    elif res < -2 ** 31:
        return -2 ** 31
    else:
        return res


# print(myAtoi(" "))


# 9. 判断回文数
def isPalindrome(x):
    if x < 0:
        return False
    elif x == 0:
        return True
    else:
        return str(x) == str(x)[::-1]


# print(isPalindrome(121))

# 10.正则表达式匹配
class Solution():
    # def isMatch(self, text, pattern):  # 回溯
    #     if not pattern:
    #         return not text
    #     first_match = bool(text) and pattern[0] in [text[0], '.']
    #     if len(pattern) >= 2 and pattern[1] == '*':
    #         return self.isMatch(text, pattern[2:]) or (first_match and self.isMatch(text[1:], pattern))
    #     else:
    #         return first_match and self.isMatch(text[1:], pattern[1:])
    def isMatch(self, text, pattern):  # 动态规划
        dp = [[False] * (len(pattern) + 1) for _ in range(len(text) + 1)]
        dp[-1][-1] = True
        for i in range(len(text), -1, -1):
            for j in range(len(pattern) - 1, -1, -1):
                first_match = i < len(text) and pattern[j] in [text[i], '.']
                if j + 1 < len(pattern) and pattern[j + 1] == '*':
                    dp[i][j] = dp[i][j + 2] or (first_match and dp[i + 1][j])
                else:
                    dp[i][j] = first_match and dp[i + 1][j + 1]
        return dp[0][0]


# s = Solution()
# print(s.isMatch('aa', 'a*'))


# 11. 盛水最多的容器
def maxArea(height):
    max_area = 0
    i, j = 0, len(height) - 1
    while i < j:
        sum = (j - i) * min(height[i], height[j])
        max_area = max(sum, max_area)
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return max_area


# print(maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))

# 12. 整数转罗马数字
def intToRoman(num: int) -> str:
    res = ''
    int = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    roman = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    for i in range(len(int)):
        while num >= int[i]:
            res += roman[i]
            num -= int[i]
    return res


# print(intToRoman(1994))

# 13. 罗马数字转整数
def romanToInt(s: str) -> int:
    res = 0
    int = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    roman = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    # for i in range(len(roman)):
    #     while s:
    #         if roman[i] in [s[0], s[0:2]]:
    #             s = s[1:] if len(roman[i]) == 1 else s[2:]
    #             res += int[i]
    #         else:
    #             break
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i + 2] in roman:
            res += int[roman.index(s[i:i + 2])]
            i += 2
        elif s[i:i + 1] in roman:
            res += int[roman.index(s[i:i + 1])]
            i += 1
    return res


# print(romanToInt('IV'))

# 14. 最长公共前缀
def longestCommonPrefix(strs) -> str:
    if not strs:
        return ''
    if len(strs) == 1:
        return strs[0]
    s = strs[0]
    res = ''
    for i in range(len(s)):
        for str in strs[1:]:
            if i < len(str) and str[i] == s[i]:
                continue
            else:
                return res
        res += s[i]
    return res


# print(longestCommonPrefix(["dlower", "flow", "flight"]))


# 15. 三数之和
def threeSum(nums):
    res = []
    if not nums and len(nums) < 3:
        return res
    nums.sort(reverse=False)
    for i in range(len(nums)):
        if nums[i] > 0:
            return res
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum == 0:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left + 1] == nums[left]:
                    left += 1
                while left < right and nums[right - 1] == nums[right]:
                    right -= 1
                left, right = left + 1, right - 1
            elif sum < 0:
                left += 1
            else:
                right -= 1
    return res


# print(threeSum([0,0,0]))


# 16.最接近的三数之和
def threeSumClosest(nums, target):
    res = 2 ** 31 - 1
    if not nums and len(nums) < 3:
        return res
    nums.sort(reverse=False)
    for i in range(len(nums)):
        left, right = i + 1, len(nums) - 1
        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if abs(sum - target) < abs(res - target):
                res = sum
            if sum > target:
                right -= 1
            elif sum < target:
                left += 1
            else:
                return res
    return res


print(threeSumClosest([1, 1, -1, -1, 3], -1))


# 42.接雨水
# def trap(height):  # 栈实现
#     ans, current = 0, 0
#     stack = []
#     while current < len(height):
#         while stack and height[current] > height[stack[-1]]:
#             top = stack.pop()
#             if not stack:
#                 break
#             distance = current - stack[-1] - 1
#             bounded_height = min(height[current], height[stack[-1]]) - height[top]
#             ans += distance * bounded_height
#         stack.append(current)
#         current += 1
#     return ans

def trap(height):
    left, right = 0, len(height) - 1
    ans = 0
    left_max, right_max = 0, 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                ans += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                ans += right_max - height[right]
            right -= 1
    return ans


# print(trap([2, 1, 0, 1, 3, 2, 1,2]))

# 43. 字符串相乘
def multiply(num1, num2):
    ans = ""
    for i in range(len(num2) - 1, -1, -1):
        res = int(num2[i])
        mul = res * int(num1) * (10 ** (len(num2) - i - 1))
        ans = addStrings(ans, str(mul))
    return ans


# 415.字符串相加
def addStrings(num1, num2):
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

# print(addStrings('123', '956'))
# print(multiply('123', '456'))
