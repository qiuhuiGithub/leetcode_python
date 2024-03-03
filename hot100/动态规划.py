# 53.最大子数组和
class Solution(object):
    def maxSubArray(self, nums):
        if not nums:
            return 0
        res = max_path = [nums[0]]
        max_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i - 1] > 0:
                max_path.append(nums[i])
                nums[i] += nums[i - 1]
            else:
                max_path = [nums[i]]
            if nums[i] > max_sum:
                max_sum = nums[i]
                res = max_path[:]  # 路径
        return max_sum, res


# 4.最长回文子串
class Solution:
    def longestPalindrome(self, s: str) -> str:
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


# 516.最长回文子序列
class Solution(object):
    def longestPalindromeSubseq(self, s):
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


# 300. 最长递增子序列
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        if not nums:
            return res
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)


# 72.编辑距离
class Solution(object):
    def minDistance(self, word1, word2):
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
                    dp[i][j] = min(min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1
        return dp[-1][-1]


# 1143.最长公共子序列
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]


# 最长公共子串
def longestCommonSubstring(text1, text2):
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


# 322. 零钱兑换
class Solution(object):
    def coinChange(self, coins, amount):
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1


# 377. 组合总和的个数
class Solution(object):
    def combinationSum4(self, nums, target):
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, target + 1):
            for num in nums:
                if num <= i:
                    dp[i] += dp[i - num]
        return dp[-1]


# 64. 最小路径和
class Solution(object):
    def minPathSum(self, grid):
        if not grid:
            return -1
        m, n = len(grid), len(grid[0])
        for i in range(1, m):
            grid[i][0] += grid[i - 1][0]
        for j in range(1, n):
            grid[0][j] += grid[0][j - 1]
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] = min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j]
        return grid[-1][-1]

# 152. 乘积最大子数组
class Solution(object):
    def maxProduct(self, nums):
        if not nums:
            return 0
        max_dp = [0 for _ in range(len(nums))]
        min_dp = [0 for _ in range(len(nums))]
        max_dp[0], min_dp[0] = nums[0], nums[0]
        for i in range(1, len(nums)):
            max_dp[i] = max(max_dp[i - 1] * nums[i], max(min_dp[i - 1] * nums[i], nums[i]))
            min_dp[i] = min(max_dp[i - 1] * nums[i], min(min_dp[i - 1] * nums[i], nums[i]))
        return max(max_dp)
