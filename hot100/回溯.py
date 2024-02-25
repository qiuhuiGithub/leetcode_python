# 46.全排列 无重复
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        if not nums:
            return res
        visits = [False] * len(nums)

        def backtrack(path):
            if len(path) == len(nums):
                res.append(path[:])
            for i in range(len(nums)):
                if visits[i]:
                    continue
                path.append(nums[i])
                visits[i] = True
                backtrack(path)
                path.pop()
                visits[i] = False

        backtrack([])
        return res

# 47.全排列 有重复
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        path = []
        nums.sort()
        used = [False for i in range(len(nums))]

        def backstrack(nums):
            if len(path) == len(nums):
                res.append(path[:])

            for i in range(len(nums)):
                if used[i]:
                    continue
                if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                    continue
                path.append(nums[i])
                used[i] = True
                backstrack(nums)
                used[i] = False
                path.pop()

        backstrack(nums)
        return res
#78.子集，无重复
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        if not nums:
            return res

        def backtrack(path, start):
            res.append(path[:])
            for i in range(start, len(nums)):
                path.append(nums[i])
                backtrack(path, i + 1)
                path.pop()

        backtrack([], 0)
        return res

# 90.子集有重复
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        if not nums:
            return res
        nums.sort()

        def backtrack(path, start):
            res.append(path[:])
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                backtrack(path, i + 1)
                path.pop()

        backtrack([], 0)
        return res

# 93. 复原IP地址
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res = []
        if not s or len(s) < 4 or len(s) > 12:
            return []
        def backtrack(i, dots, curIP):
            if dots == 4 and i == len(s):
                res.append(curIP[:-1])
                return
            if dots > 4:
                return
            for j in range(i, min(i+3, len(s))):
                if int(s[i:j+1]) < 256 and (s[i] != '0' or i ==j):
                    backtrack(j+1, dots+1, curIP + s[i:j+1] + '.')
        backtrack(0,0,"")
        return res

# 22. 括号生成
class Solution(object):
    def generateParenthesis(self, n):
        res = []
        if n == 0:
            return res

        def backtrack(path, left, right):
            if len(path) == 2 * n:
                res.append(path)
            if left < n:
                path += '('
                backtrack(path, left + 1, right)
                path = path[:-1]
            if right < left:
                path += ')'
                backtrack(path, left, right + 1)
                path = path[:-1]

        backtrack("", 0, 0)
        return res

# 39. 组合总和 可重复使用
class Solution(object):
    def combinationSum(self, candidates, target):
        candidates.sort()
        res = []
        def backtrack(path, target, start):
            if target < 0:
                return
            if target == 0:
                res.append(path[:])
                return

            for i in range(start, len(candidates)):
                path.append(candidates[i])
                backtrack(path, target - candidates[i], i)
                path.pop()

        backtrack([], target, 0)
        return res

# 40. 组合总和 只能用一次
class Solution(object):
    def combinationSum2(self, candidates, target):
        candidates.sort()
        res = []

        def backtrack(path, target, start):
            if target < 0:
                return
            if target == 0:
                res.append(path[:])
                return

            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                backtrack(path, target - candidates[i], i + 1)
                path.pop()

        backtrack([], target, 0)
        return res

#216. 组合总和III
class Solution(object):
    def combinationSum3(self, k, n):
        res = []

        def backtrack(path, target, start):
            if target < 0:
                return
            if target == 0 and len(path) == k:
                res.append(path[:])
                return

            for i in range(start, 10):
                path.append(i)
                backtrack(path, target - i, i + 1)
                path.pop()

        backtrack([], n, 1)
        return res