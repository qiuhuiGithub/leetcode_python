# 42.接雨水
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left, right = 0, len(height) - 1
        ans = 0
        left_max, right_max = 0, 0
        while left < right:
            if height[left] < height[right]:
                if height[left] < left_max:
                    ans += left_max - height[left]
                else:
                    left_max = height[left]
                left += 1
            else:
                if height[right] < right_max:
                    ans += right_max - height[right]
                else:
                    right_max = height[right]
                right -= 1
        return ans

# 209. 长度最小的子数组
class Solution(object):
    def minSubArrayLen(self, target, nums):
        min_len = float('inf')
        if not nums:
            return 0
        slow, fast = 0, 0
        target_sum = 0
        while fast < len(nums):
            target_sum += nums[fast]
            while target_sum >= target:
                min_len = min(min_len, fast - slow + 1)
                target_sum -= nums[slow]
                slow += 1
            fast += 1
        return min_len if min_len != float('inf') else 0