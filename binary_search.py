# encoding=utf-8
# 标准二分查找
def binary_search(nums, target):
    if not nums:
        return -1
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left if nums[left] == target else -1


# 寻找左边界
def binary_search_left_bound(nums, target):
    if not nums:
        return -1
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            right = mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left if nums[left] == target else -1


# 寻找右边界
def binary_search_right_bound(nums, target):
    if not nums:
        return -1
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left + 1) // 2
        if nums[mid] == target:
            left = mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left if nums[left] == target else -1


# 33. 搜索旋转排序数组
def search(nums, target):
    if not nums:
        return -1
    left, right = 0, len(nums) - 1
    mid = left + (right - left) // 2
    while left <= right:
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target <= nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] <= target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
        mid = left + (right - left) // 2
    return -1


# print(search([4, 5, 6, 1, 2, 3], 3))


# 34.在排序数组中查找元素的第一个和最后一个位置
def searchRange(nums, target):  # 二分
    if not nums:
        return [-1, -1]
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            right = mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    if nums[left] != target:
        return [-1, -1]
    start, right = left, len(nums) - 1
    while left < right:
        mid = left + (right - left + 1) // 2
        if nums[mid] == target:
            left = mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    end = left
    return [start, end]


# 69.x的平方根
def mySqrt(x):
    """
    :type x: int
    :rtype: int
    """
    if x == 0:
        return 0
    left, right, ans = 0, x, -1
    while left <= right:
        mid = left + (right - left) // 2
        if mid * mid <= x:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
    return ans


# print(mySqrt(2147395599))


# 74. 搜索二维矩阵
def searchMatrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    for i in range(len(matrix)):
        if matrix[i][0] <= target <= matrix[i][-1]:
            low, high = 0, len(matrix[i]) - 1
            while low <= high:
                mid = low + (high - low) // 2
                if matrix[i][mid] == target:
                    return True
                if target < matrix[i][mid]:
                    high = mid - 1
                else:
                    low = mid + 1
    return False


# 81. 搜索旋转排序数组 II
def search(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: bool
    """
    if not nums:
        return False
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return True
        # if nums[left] == nums[mid]:
        #     left += 1
        #     continue
        if nums[left] <= nums[mid]:
            if nums[left] <= target <= nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] <= target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return False


print(search([1, 3, 1, 1, 1], 3))


# 153. 寻找旋转排序数组中的最小值
def findMin(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return -1
    if len(nums) == 1:
        return nums[0]
    if nums[0] < nums[-1]:
        return nums[0]
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if nums[mid] > nums[mid + 1]:
            return nums[mid + 1]
        elif nums[mid - 1] > nums[mid]:
            return nums[mid]
        if nums[low] < nums[mid]:
            low = mid + 1
        else:
            high = mid - 1


# 215. 数组中的第K个最大元素
def findKthLargest(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """

    def partition(nums, left, right):
        pivot = nums[left]
        while left < right:
            while left < right and nums[right] >= pivot:
                right -= 1
            nums[left] = nums[right]
            while left < right and nums[left] < pivot:
                left += 1
            nums[right] = nums[left]
        nums[left] = pivot
        return left

    # target_k = len(nums) - k
    target_k = k
    low, high = 0, len(nums) - 1
    if target_k == len(nums):
        return nums
    while True:
        index = partition(nums, low, high)
        if index == target_k:
            return nums[:index]
        elif index < target_k:
            low = index + 1
        else:
            high = index - 1

# print(findKthLargest([0, 0, 2, 3, 2, 1, 1, 2, 0, 4], 10))
