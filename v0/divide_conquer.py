# encoding=utf-8
# 169.多数元素
def majorityElement(nums):
    """
    :type nums: List[int]
    :rtype: int
    """

    # def partition(nums, left, right):
    #     pivot = nums[left]
    #     while left < right:
    #         while left < right and nums[right] >= pivot:
    #             right -= 1
    #         nums[left] = nums[right]
    #         while left < right and nums[left] < pivot:
    #             left += 1
    #         nums[right] = nums[left]
    #     nums[left] = pivot
    #     return left
    #
    # middle = len(nums) // 2
    # index = partition(nums, 0, len(nums) - 1)
    # while index != middle:
    #     if index < middle:
    #         index = partition(nums, index + 1, len(nums) - 1)
    #     else:
    #         index = partition(nums, 0, index - 1)
    # return nums[index]
    res, cnt = 0, 0
    for num in nums:
        if cnt == 0:
            res = num
            cnt += 1
        else:
            cnt = cnt + 1 if res == num else cnt - 1
    return res