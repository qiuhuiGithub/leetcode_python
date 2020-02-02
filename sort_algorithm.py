# encoding=utf-8


def bubble_sort(nums):
    """
    冒泡排序
    :param nums:
    :return:
    """
    if not nums:
        return None
    n = len(nums)
    for i in range(n):
        for j in range(n - 1 - i):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums


def select_sort(nums):
    if not nums:
        return
    n = len(nums)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if nums[j] < nums[min_index]:
                min_index = j
        nums[min_index], nums[i] = nums[i], nums[min_index]
    return nums


def insert_sort(nums):
    if not nums:
        return
    n = len(nums)
    for i in range(n - 1):
        current = nums[i + 1]
        pre_index = i
        while pre_index >= 0 and current < nums[pre_index]:
            nums[pre_index + 1] = nums[pre_index]
            pre_index -= 1
        nums[pre_index + 1] = current
    return nums


def shell_sort(nums):
    if not nums:
        return
    # todo
    return nums


if __name__ == "__main__":
    nums = [3, 5, 7, 2, 4, 6, 8]
    # print(bubble_sort(nums))
    # print(select_sort(nums))
    # print(insert_sort(nums))
    # print(shell_sort(nums))
