# 15. 三数之和
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return []
        nums.sort()
        res = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left, right = i + 1, len(nums) - 1
            while left < right:
                target = nums[i] + nums[left] + nums[right]
                if target == 0:
                    res.append([i, left, right])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif target < 0:
                    left += 1
                else:
                    right -= 1
        return res


# 16.最接近的三数之和
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        res = float('inf')
        if not nums:
            return res
        nums.sort()
        length = len(nums)
        for i in range(length):
            left, right = i + 1, length - 1
            while left < right:
                s = nums[i] + nums[left] + nums[right]
                if abs(s - target) < abs(res - target):
                    res = s
                if s == target:
                    return s
                elif s < target:
                    left += 1
                else:
                    right -= 1
        return res


# 18.四数之和
class Solution:
    def fourSum(self, nums, target):
        res = []
        if not nums or len(nums) < 4:
            return res
        nums.sort(reverse=False)
        for i in range(len(nums) - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, len(nums) - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                k, l = j + 1, len(nums) - 1
                while k < l:
                    sum = nums[i] + nums[j] + nums[k] + nums[l]
                    if sum == target:
                        res.append([nums[i], nums[j], nums[k], nums[l]])
                        while k < l and nums[k + 1] == nums[k]:
                            k += 1
                        while k < l and nums[l - 1] == nums[l]:
                            l -= 1
                        k, l = k + 1, l - 1
                    elif sum < target:
                        k += 1
                    else:
                        l -= 1
        return res


# 54. 螺旋矩阵
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        ans = []
        if not matrix:
            return ans
        row, col = len(matrix), len(matrix[0])
        visit = [[False] * col for _ in range(row)]
        dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        r, c, d = 0, 0, 0
        for _ in range(row * col):
            visit[r][c] = True
            ans.append(matrix[r][c])
            if 0 <= r + dir[d][0] < row and 0 <= c + dir[d][1] < col and not visit[r + dir[d][0]][c + dir[d][1]]:
                r, c = r + dir[d][0], c + dir[d][1]
            else:
                d = (d + 1) % 4
                r, c = r + dir[d][0], c + dir[d][1]
        return ans

# 31. 下一个排列
class Solution(object):
    def nextPermutation(self, nums):
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1:] = nums[i + 1:][::-1]

# 41. 缺失的第一个正数
class Solution(object):
    def firstMissingPositive(self, nums):
        if not nums or 1 not in nums:
            return 1
        if nums == [1]:
            return 2
        n = len(nums)
        for i in range(n):
            if nums[i] > n or nums[i] <= 0:
                nums[i] = 1
        for i in range(n):
            idx = abs(nums[i])
            if idx == n:
                nums[0] = -abs(nums[0])
            else:
                nums[idx] = -abs(nums[idx])
        for i in range(1, n):
            if nums[i] > 0:
                return i
        if nums[0] > 0:
            return n
        return n + 1