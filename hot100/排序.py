# encoding=utf-8

"""
排序算法    平均时间复杂度   平均空间复杂度   稳定性
冒泡排序    O(n²)           O(1)          稳定
选择排序    O(n²)           O(1)          不稳定
插入排序    O(n²)           O(1)          稳定
希尔排序    O(nlogn)        O(1)          不稳定
归并排序    O(nlogn)        O(n)          稳定
快速排序    O(nlogn)        O(logn)       不稳定
堆排序      O(nlogn)        O(1)          不稳定
"""


def bubble_sort(nums):
    """
    :param nums:
    :return:
    """
    if not nums:
        return
    n = len(nums)
    for i in range(n):
        for j in range(n - 1 - i):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]


def select_sort(nums):
    """
    选择排序
    :param nums:
    :return:
    """
    if not nums:
        return
    n = len(nums)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if nums[j] < nums[min_index]:
                min_index = j
        nums[min_index], nums[i] = nums[i], nums[min_index]


def insert_sort(nums):
    """
    插入排序
    :param nums:
    :return:
    """
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


def shell_sort(nums):
    """
    希尔排序
    :param nums:
    :return:
    """
    if not nums:
        return
    n = len(nums)
    tmp, gap = 0, n // 2
    while gap > 0:
        for i in range(gap, n):
            tmp = nums[i]
            pre_index = i - gap
            while pre_index >= 0 and nums[pre_index] > tmp:
                nums[pre_index + gap] = nums[pre_index]
                pre_index -= gap
            nums[pre_index + gap] = tmp
        gap //= 2


def merge_sort(nums):
    """
    归并排序
    :param nums:
    :return:
    """
    if len(nums) <= 1:
        return nums
    middle = len(nums) // 2
    left = merge_sort(nums[:middle])
    right = merge_sort(nums[middle:])
    merge(left, right)


def merge(left, right):
    """
    归并排序合并函数
    :param left:
    :param right:
    :return:
    """
    res = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    if i == len(left):
        res.extend(right[j:])
    if j == len(right):
        res.extend(left[i:])
    return res


def quick_sort(nums, low, high):
    """
    快速排序
    :param nums:
    :param low:
    :param high:
    :return:
    """
    if low < high:
        index = partition(nums, low, high)
        quick_sort(nums, low, index - 1)
        quick_sort(nums, index + 1, high)


def partition(nums, left, right):
    """
    快排划分函数
    :param nums:
    :param left:
    :param right:
    :return:
    """
    piovt = nums[left]
    while left < right:
        while left < right and nums[right] >= piovt:
            right -= 1
        nums[left] = nums[right]
        while left < right and nums[left] < piovt:
            left += 1
        nums[right] = nums[left]
    nums[left] = piovt
    return left


def heapify(nums, n, i):
    """
    调整堆结构
    :param nums:
    :param n:
    :param i:
    :return:
    """
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and nums[i] < nums[l]:
        largest = l
    if r < n and nums[largest] < nums[r]:
        largest = r
    if largest != i:
        nums[i], nums[largest] = nums[largest], nums[i]
        heapify(nums, n, largest)


def heap_sort(nums):
    if not nums:
        return
    n = len(nums)
    for i in range(n, -1, -1):
        heapify(nums, n, i)
    for i in range(n - 1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]
        heapify(nums, i, 0)


if __name__ == "__main__":
    nums = [5, 3, 7, 2, 4, 6]
    # bubble_sort(nums)
    # select_sort(nums)
    # insert_sort(nums)
    # shell_sort(nums)
    # merge_sort(nums)
    quick_sort(nums, 0, len(nums) - 1)
    # heap_sort(nums)
    print(nums)

