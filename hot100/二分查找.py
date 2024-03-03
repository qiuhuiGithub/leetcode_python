# 33.搜索旋转排序数组
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
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

# 162. 寻找峰值
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        def get(i):
            if i == -1 or i == n:
                return float('-inf')
            return nums[i]

        left, right, ans = 0, n - 1, -1
        while left <= right:
            mid = (left + right) // 2
            if get(mid - 1) < get(mid) > get(mid + 1):
                ans = mid
                break
            elif get(mid) < get(mid + 1):
                left = mid + 1
            else:
                right = mid - 1
        return ans