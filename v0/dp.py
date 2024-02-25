# encoding=utf-8
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


# 512. 最长回文子序列
def longestPalindromeSubseq(s):
    """
    :type s: str
    :rtype: int
    """
    if not s:
        return 0
    length = len(s)
    dp = [[0 for _ in range(length)] for _ in range(length)]
    for i in range(length - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, length):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][length - 1]


# print(longestPalindromeSubseq('bbab'))


# 10.正则表达式匹配
class Solution():
    # def isMatch(self, text, pattern):  # 回溯
    #     if not pattern:
    #         return not text
    #     first_match = bool(text) and pattern[0] in [text[0], "."]
    #     if len(pattern) >= 2 and pattern[1] == "*":
    #         return self.isMatch(text, pattern[2:]) or (first_match and self.isMatch(text[1:], pattern))
    #     else:
    #         return first_match and self.isMatch(text[1:], pattern[1:])
    # 自顶向下
    # dp[i][j] = dp[i][j+2] || (p[j]==t[i] or p[i]=='.') & dp[i+1][j] if p[j+1] == '*'
    # dp[i][j] = dp[i+1][j+1] & (p[j]==t[i] or p[i]=='.') if p[j+1] != '*'
    # def isMatch(self, text, pattern):  # 动态规划
    #     dp = [[False] * (len(pattern) + 1) for _ in range(len(text) + 1)]
    #     dp[-1][-1] = True
    #     for i in range(len(text), -1, -1):
    #         for j in range(len(pattern) - 1, -1, -1):
    #             first_match = i < len(text) and pattern[j] in [text[i], "."]
    #             if j + 1 < len(pattern) and pattern[j + 1] == "*":
    #                 dp[i][j] = dp[i][j + 2] or (first_match and dp[i + 1][j])
    #             else:
    #                 dp[i][j] = first_match and dp[i + 1][j + 1]
    #     return dp[0][0]
    # dp[i][j] = i > 0 and dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '.') if p[j-1]!= '*'
    # dp[i][j] = dp[i][j - 2] or (i > 0 and (s[i - 1] == p[j - 2] or p[j - 2] == '.') and dp[i - 1][j]) if p[j-1] == '*'
    # 自底向上
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
        dp[0][0] = True
        for i in range(0, len(s)+1):
            for j in range(1, len(p)+1):
                if p[j-1] == '*':
                    dp[i][j] = dp[i][j - 2] or (i > 0 and (s[i-1] == p[j-2] or p[j-2] == '.') and dp[i - 1][j]);
                else:
                    dp[i][j] = i > 0 and dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '.')
        return dp[-1][-1]


solution = Solution()


# print(solution.isMatch('bb', 'b*'))


# 1143.最长公共子序列
def longestCommonSubsequence(text1, text2):
    """
    :type text1: str
    :type text2: str
    :rtype: int
    """
    if not text1 or not text2:
        return 0
    len1, len2 = len(text1), len(text2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


# 最长公共子串
def longestCommonSubstring(text1, text2):
    if not text1 or not text2:
        return 0
    len1, len2 = len(text1), len(text2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    max_len = 0
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0
    return max_len


# 44.通配符匹配
def isMatch(s, p):
    """
    dp[i][j] = dp[i-1][j-1] if s[i] == p[j] or p[j] == '?'
    dp[i][j] = dp[i-1][j] || dp[i][j-1] if p[j] == '*'
    :param s:
    :param p:
    :return:
    """
    m, n = len(s), len(p)
    dp = [[False for _ in range(n + 1)] for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] and p[j - 1] == '*'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == p[j - 1] or p[j - 1] == '?':
                dp[i][j] = dp[i - 1][j - 1]
            if p[j - 1] == '*':
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
    return dp[m][n]


# 62. 不同路径
def uniquePaths(m, n):
    """
    :type m: int
    :type n: int
    :rtype: int
    """
    if not m or not n:
        return -1
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


#  63. 不同路径II
def uniquePathsWithObstacles(obstacleGrid):
    """
    :type obstacleGrid: List[List[int]]
    :rtype: int
    """
    if not obstacleGrid:
        return -1
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    row_flag, col_flag = True, True
    for i in range(m):
        if obstacleGrid[i][0] == 0 and row_flag:
            dp[i][0] = 1
        else:
            dp[i][0] = 0
            row_flag = False

    for j in range(n):
        if obstacleGrid[0][j] == 0 and col_flag:
            dp[0][j] = 1
        else:
            dp[0][j] = 0
            col_flag = False
    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] == 1:
                dp[i][j] = 0
            elif obstacleGrid[i - 1][j] == 0 and obstacleGrid[i][j - 1] == 0:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
            elif obstacleGrid[i - 1][j] == 1:
                dp[i][j] = dp[i][j - 1]
            elif obstacleGrid[i][j - 1] == 1:
                dp[i][j] = dp[i - 1][j]
    return dp[-1][-1]


# 64. 最小路径和
def minPathSum(grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    if not grid:
        return -1
    row, col = len(grid), len(grid[0])
    dp = [[0] * col for _ in range(row)]
    dp[0][0] = grid[0][0]
    for i in range(1, row):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, col):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, row):
        for j in range(1, col):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    return dp[-1][-1]


# 72.编辑距离
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], min(dp[i][j - 1], dp[i - 1][j]))
    return dp[m][n]


# 120.三角形最小路径和
def minimumTotal(triangle):
    """
    :type triangle: List[List[int]]
    :rtype: int
    """
    if not triangle:
        return 0
    for i in range(len(triangle) - 2, -1, -1):
        for j in range(len(triangle[i])):
            triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1])
    return triangle[0][0]


# 139. 单词拆分
def wordBreak(s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    if not s or not wordDict:
        return False
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(0, i):
            if dp[j] and s[j:i] in wordDict:
                dp[i] = True
                break
    return dp[-1]


def wordBreak(s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    if not s or not wordDict:
        return 0
    dp = [0] * (len(s) + 1)
    dp[0] = 1
    for i in range(1, len(s) + 1):
        for j in range(0, i):
            if dp[j] and s[j:i] in wordDict:
                dp[i] = dp[i] + dp[j]
    return dp[-1]


# print(wordBreak('leetcode', {'lee', 't', 'leet', 'code', 'leetcode'}))


# 152. 乘积最大的子数组
def maxProduct(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return 0
    length = len(nums)
    max_dp, min_dp = [0 for _ in range(length)], [0 for _ in range(length)]
    max_dp[0] = nums[0]
    min_dp[0] = nums[0]
    for i in range(1, length):
        max_dp[i] = max(max_dp[i - 1] * nums[i], max(min_dp[i - 1] * nums[i], nums[i]))
        min_dp[i] = min(min_dp[i - 1] * nums[i], min(max_dp[i - 1] * nums[i], nums[i]))
    return max(max_dp)


# print(maxProduct([5, 6, -3, 4, 3]))


# 198. 打家劫舍
def rob(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    nums[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        nums[i] = max(nums[i - 1], nums[i - 2] + nums[i])
    return nums[-1]


def rob(nums):
    """
    :type nums: List[int]
    :rtype: int, List[int]路径
    """
    if not nums:
        return 0, []
    if len(nums) == 1:
        return nums[0], [0]
    path = [[] for _ in range(len(nums))]
    path[0] = [0]
    if nums[0] < nums[1]:
        path[1] = [1]
    else:
        path[1] = [0]
    nums[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        if nums[i - 1] < nums[i - 2] + nums[i]:
            nums[i] = nums[i - 2] + nums[i]
            path[i] = path[i - 2] + [i]
        else:
            nums[i] = nums[i - 1]
            path[i] = path[i - 1]
    return nums[-1], path[-1]


# print(rob([2, 6, 3]))


# 213. 打家劫舍II
def rob_2(nums):
    """
    :type nums: List[int]
    :rtype: int
    """

    def rob_helper(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        nums[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            nums[i] = max(nums[i - 1], nums[i - 2] + nums[i])
        return nums[-1]

    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    return max(rob_helper(nums[1:]), rob_helper(nums[:-1]))


# 221. 最大正方形
def maximalSquare(matrix):
    """
    :type matrix: List[List[str]]
    :rtype: int
    """
    if not matrix:
        return 0
    row, col = len(matrix), len(matrix[0])
    dp = [[0] * (col + 1) for _ in range(row + 1)]
    max_len = 0
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if matrix[i - 1][j - 1] == '0':
                continue
            dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1
            max_len = max(max_len, dp[i][j])
    return max_len * max_len


# 300. 最长上升子序列
def lengthOfLIS(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return 0
    dp = [1] * (len(nums))
    for i in range(1, len(nums)):
        cnt = 1
        for j in range(i):
            if nums[i] > nums[j]:
                cnt = max(cnt, dp[j] + 1)
        dp[i] = cnt
    return max(dp)


# 0-1背包
def knapsack(N, W, wt, val):
    dp = [[0 for _ in range(W + 1)] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for w in range(1, W + 1):
            if w - wt[i - 1] < 0:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - wt[i - 1]] + val[i - 1])
    return dp[N][W]


# print(knapsack(3, 4, [2, 2, 2], [4, 2, 3]))

# 416. 分割等和子集
def canPartition(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    if not nums:
        return False
    arr_sum = sum(nums)
    if arr_sum % 2 != 0:
        return False
    N, W = len(nums), arr_sum // 2
    dp = [[False for _ in range(W + 1)] for _ in range(N + 1)]
    for i in range(N + 1):
        dp[i][0] = True
    for i in range(1, N + 1):
        for j in range(1, W + 1):
            if j < nums[i - 1]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] | dp[i - 1][j - nums[i - 1]]
    return dp[-1][-1]


# print(canPartition([1, 1, 2]))

# 面试题08.11. 硬币
def waysToChange(n):
    """
    :type n: int
    :rtype: int
    """
    MAX = 1000000007
    if n < 0:
        return 0
    dp = [0] * (n + 1)
    dp[0] = 1
    coins = [25, 10, 5, 1]

    for coin in coins:
        for i in range(coin, n + 1):
            dp[i] += dp[i - coin]
    return dp[-1] % MAX


# print(waysToChange(10))

# 322. 零钱兑换
def coinChange(coins, amount):
    """
    :type coins: List[int]
    :type amount: int
    :rtype: int
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[-1] if dp[-1] != float('inf') else -1


# print(coinChange([2], 11))

# 494. 目标和
def findTargetSumWays(nums, S):
    """
    :type nums: List[int]
    :type S: int
    :rtype: int
    """
    length, max_sum = len(nums), sum(nums)
    if S > max_sum or S < -max_sum:
        return 0
    dp = [{} for _ in range(length)]
    dp[0][nums[0]] = 1
    dp[0][-nums[0]] = 1
    if nums[0] == 0:
        dp[0][0] = 2
    for i in range(1, length):
        for j in range(-max_sum, max_sum + 1):
            dp[i][j] = dp[i - 1].get(j - nums[i], 0) + dp[i - 1].get(j + nums[i], 0)
    return dp[-1][S]


print(findTargetSumWays([1, 0], 1))


# 518.零钱兑换
def change(amount, coins):
    """
    :type amount: int
    :type coins: List[int]
    :rtype: int
    """
    if amount == 0 and not coins:
        return 1
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[-1]

# print(change(0, []))
