# encoding=utf-8

# 3. 无重复字符的最长子串
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


# 6. z字形变换
def convert(s, numRows):
    if numRows == 1:
        return s
    res = [""] * min(numRows, len(s))
    cur_row = 0
    go_down = -1
    for c in s:
        res[cur_row] += c
        if cur_row == 0 or cur_row == numRows - 1:
            go_down = -go_down
        cur_row += go_down
    return "".join(res)


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


# 8.字符串转数字
def myAtoi(str):
    str = str.strip()
    if not str:
        return 0
    if str[0] not in ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        return 0
    minus = 1
    if str[0] == "-":
        minus = -1
        str = str[1:]
    elif str[0] == "+":
        str = str[1:]
    for i in range(len(str)):
        if str[i] != "0":
            str = str[i:]
            break
    if not str:
        return 0
    res = ""
    for char in str:
        if char in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
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


# 12. 整数转罗马数字
def intToRoman(num: int) -> str:
    res = ""
    int = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    roman = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    for i in range(len(int)):
        while num >= int[i]:
            res += roman[i]
            num -= int[i]
    return res


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


# 14. 最长公共前缀
def longestCommonPrefix(strs) -> str:
    if not strs:
        return ""
    if len(strs) == 1:
        return strs[0]
    s = strs[0]
    res = ""
    for i in range(len(s)):
        for str in strs[1:]:
            if i < len(str) and str[i] == s[i]:
                continue
            else:
                return res
        res += s[i]
    return res


print(longestCommonPrefix(['flower', 'floe', 'fl']))


# 28. 实现strStr()
def strStr(haystack, needle):
    if not needle:
        return 0
    for i in range(len(haystack)):
        if needle[0] == haystack[i] and haystack[i:i + len(needle)] == needle:
            return i
    return -1


# 29. 两数相除
def divide(dividend, divisor):
    if dividend == 0:
        return 0
    if dividend == -2 ** 31 and divisor == -1:
        return 2 ** 31 - 1

    negative = 1 if dividend ^ divisor >= 0 else -1
    dividend, divisor = abs(dividend), abs(divisor)
    res = 0
    for i in range(31, -1, -1):
        if (dividend >> i) >= divisor:
            res += 1 << i
            dividend -= divisor << i
    return negative * res


# 30.串联所有单词的子串
def findSubstring(s, words):
    words.sort()
    res = []
    if not s or not words:
        return res
    word_len = len(words[0])
    list_len = len(words)
    substr_len = word_len * list_len
    for i in range(0, len(s) - substr_len + 1):
        sub_str = s[i:i + substr_len]
        tmp = []
        for j in range(0, len(sub_str), word_len):
            tmp.append(sub_str[j:j + word_len])
        tmp.sort()
        if tmp == words:
            res.append(i)
    return res


# 31.下一个排列
def nextPermutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = len(nums) - 1
        while j >= 0 and nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1:] = nums[i + 1:][::-1]


# 38.外观数列
def countAndSay(n):
    s = '1'
    while n > 1:
        s += '0'
        tmp = ''
        cnt = 1
        for i in range(len(s) - 1):
            if s[i] == s[i + 1]:
                cnt += 1
            else:
                tmp += str(cnt) + str(s[i])
                cnt = 1
        s = tmp
        n -= 1
    return s


# 43. 字符串相乘
def multiply(num1, num2):
    ans = ""
    for i in range(len(num2) - 1, -1, -1):
        res = int(num2[i])
        mul = res * int(num1) * (10 ** (len(num2) - i - 1))
        ans = addStrings(ans, str(mul))
    return ans


# 58. 最后一个单词的长度
def lengthOfLastWord(s):
    """
    :type s: str
    :rtype: int
    """
    if not s.strip():
        return 0
    cnt = 0
    s = s[::-1].strip()
    for i in range(len(s)):
        if s[i] != ' ':
            cnt += 1
        else:
            break
    return cnt


print(lengthOfLastWord('abc ce'))


# 67.二进制求和
def addBinary(a, b):
    """
    :type a: str
    :type b: str
    :rtype: str
    """
    length = max(len(a), len(b))
    a, b = '0' * (length - len(a)) + a, '0' * (length - len(b)) + b
    carry = 0
    res = ''
    for i in range(length - 1, -1, -1):
        sum = (int(a[i]) + int(b[i]) + carry) % 2
        carry = (int(a[i]) + int(b[i]) + carry) // 2
        res += str(sum)
    if carry:
        res += str(carry)
    return res[::-1]


print(addBinary('11', '01'))


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


# 49.字母异位词分组
def groupAnagrams(strs):
    if not strs:
        return []
    dic = {}
    for word in strs:
        tmp = ''.join(sorted(word))
        if tmp in dic.keys():
            dic[tmp].append(word)
        else:
            dic[tmp] = [word]
    return list(dic.values())


# 71. 路径简化
def simplifyPath(path):
    """
    :type path: str
    :rtype: str
    """
    res = []
    strs = path.split('/')
    for str in strs:
        if str == '..':
            if res:
                res.pop()
        elif str != '' and str != '.':
            res.append(str)
    if not res:
        return '/'
    return '/' + '/'.join(res)


# print(simplifyPath('/../'))

# 76. 最小覆盖子串
def minWindow(s, t):
    """
    :param s:
    :param t:
    :return:
    """
    import collections
    need = collections.defaultdict(int)
    for c in t:
        need[c] += 1
    needCnt = len(t)
    i = 0
    res = (0, float('inf'))
    for j, c in enumerate(s):
        if need[c] > 0:
            needCnt -= 1
        need[c] -= 1
        if needCnt == 0:  # 步骤一：滑动窗口包含了所有T元素
            while True:  # 步骤二：增加i，排除多余元素
                c = s[i]
                if need[c] == 0:
                    break
                need[c] += 1
                i += 1
            if j - i < res[1] - res[0]:  # 记录结果
                res = (i, j)
            need[s[i]] += 1  # 步骤三：i增加一个位置，寻找新的满足条件滑动窗口
            needCnt += 1
            i += 1
    return '' if res[1] > len(s) else s[res[0]:res[1] + 1]  # 如果res始终没被更新过，代表无满足条件的结果


print(minWindow('DOABECODEBANC', 'ABC'))
