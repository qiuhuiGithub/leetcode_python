# encoding=utf-8
# 1. two sum
def twoSum(nums, target):
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]


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


# 18. 四数之和
def fourSum(nums, target):
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


# 26. 删除排序数组中的重复项
def removeDuplicates(nums):
    if not nums:
        return 0
    i = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    return i + 1


# 27. 移除元素
def removeElement(nums, val):
    if not nums:
        return 0
    i = 0
    for j in range(len(nums)):
        if nums[j] != val:
            nums[i] = nums[j]
            i += 1
    return i


# 35. 搜索插入位置
def searchInsert(nums, target):
    for i in range(len(nums)):
        if nums[i] >= target:
            return i
    return len(nums)


# 41.缺失的第一个正数
def firstMissingPositive(nums):
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


# 42.接雨水
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


# 48.旋转图像
def rotate(matrix):  # 先转置，后翻转
    if not matrix:
        return
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for i in range(n):
        for j in range(n // 2):
            matrix[i][j], matrix[i][n - 1 - j] = matrix[i][n - 1 - j], matrix[i][j]
    return matrix


# 53. 最大子序和
def maxSubArray(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return 0
    max_sum = nums[0]
    for i in range(1, len(nums)):
        if nums[i - 1] > 0:
            nums[i] += nums[i - 1]
        max_sum = max(max_sum, nums[i])
    return max_sum


# 54. 螺旋矩阵
def spiralOrder(matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    ans = []
    if not matrix:
        return ans
    row, col = len(matrix), len(matrix[0])
    visit = [[False] * col for _ in range(row)]
    dir = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    r = c = d = 0
    for _ in range(row * col):
        visit[r][c] = True
        ans.append(matrix[r][c])
        if 0 <= r + dir[d][0] < row and 0 <= c + dir[d][1] < col and not visit[r + dir[d][0]][c + dir[d][1]]:
            r, c = r + dir[d][0], c + dir[d][1]
        else:
            d = (d + 1) % 4
            r, c = r + dir[d][0], c + dir[d][1]
    return ans


print(spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


# 218. 天际线问题
def getSkyline(buildings):
    """
    :type buildings: List[List[int]]
    :rtype: List[List[int]]
    """

    res = []
    if not buildings:
        return res
    height = []
    for building in buildings:
        height.append([building[0], -building[2]])
        height.append([building[1], building[2]])
    height.sort()

    heap = [0]
    prev = 0
    for h in height:
        if h[1] < 0:
            heap.append(-h[1])
        else:
            heap.remove(h[1])
        cur = max(heap)
        if prev != cur:
            res.append([h[0], cur])
            prev = cur

    final_res = [[height[0][0], 0]]
    for i in range(len(res) - 1):
        final_res.append(res[i])
        final_res.append([res[i + 1][0], res[i][1]])
    final_res.append(res[-1])
    return final_res

# print(getSkyline([[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]))
