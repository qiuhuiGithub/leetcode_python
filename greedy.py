# encoding=utf-8
# 45.跳跃游戏II
def jump(nums):
    end, max_pos, step = 0, 0, 0
    for i in range(len(nums) - 1):
        max_pos = max(max_pos, nums[i] + i)
        if i == end:
            end = max_pos
            step += 1
    return step

# 55.跳跃游戏
def canJump(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    n, rightmost = len(nums), 0
    for i in range(n):
        if i <= rightmost:
            rightmost = max(rightmost, i + nums[i])
            if rightmost >= n - 1:
                return True
    return False

# 121. 买卖股票的最佳时机
def maxProfit(prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    min_price = 2 ** 31 - 1
    max_profit = 0
    for price in prices:
        max_profit = max(price - min_price, max_profit)
        min_price = min(min_price, price)
    return max_profit


# 122. 买卖股票的最佳时机 II
def maxProfit(prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    max_profit = 0
    for i in range(1, len(prices)):
        if prices[i] - prices[i - 1] > 0:
            max_profit += prices[i] - prices[i - 1]
    return max_profit


# 123. 买卖股票的最佳时机 II
def maxProfit(prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    first_buy, first_sell = -2 ** 31, 0
    second_buy, second_sell = -2 ** 31, 0
    for price in prices:
        first_buy = max(first_buy, -price)
        first_sell = max(first_sell, price + first_buy)
        second_buy = max(second_buy, first_sell - price)
        second_sell = max(second_sell, price + second_buy)
    return second_sell

# 134.加油站
def canCompleteCircuit(gas, cost):
    """
    :type gas: List[int]
    :type cost: List[int]
    :rtype: int
    """
    if not gas or not cost or len(gas) != len(cost):
        return -1
    start, sum, total = 0, 0, 0
    for i in range(len(gas)):
        sum += gas[i] - cost[i]
        total += gas[i] - cost[i]
        if sum < 0:
            start = i + 1
            sum = 0
    return start if total >= 0 else -1

# 135. 分发糖果
def candy(ratings):
    """
    :type ratings: List[int]
    :rtype: int
    """
    if not ratings:
        return 0
    candy = [1] * len(ratings)
    for i in range(len(ratings) - 1):
        if ratings[i + 1] > ratings[i]:
            candy[i + 1] = candy[i] + 1
    for i in range(len(ratings) - 1, 0, -1):
        if ratings[i - 1] > ratings[i] and candy[i - 1] <= candy[i]:
            candy[i - 1] = candy[i] + 1
    return sum(candy)