# 215数据中第K大的元素
class Solution(object):
    def findKthLargest(self, nums, k):
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

        target_k = len(nums) - k
        low, high = 0, len(nums) - 1
        while True:
            index = partition(nums, low, high)
            if index == target_k:
                return nums[index]
            elif index < target_k:
                low = index + 1
            else:
                high = index - 1