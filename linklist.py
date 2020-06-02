# encoding=utf-8
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 2. add two numbers
def addTwoNumbers(l1, l2):
    head = ListNode(0)
    result = head
    add = 0
    while l1 or l2:
        x = l1.val if l1 else 0
        y = l2.val if l2 else 0
        sum = x + y + add
        add = sum // 10
        result.next = ListNode(sum % 10)
        result = result.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    if add != 0:
        result.next = ListNode(1)
    return head.next

# 19. 删除链表的倒数第n个节点
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    start = ListNode(0)
    fast = slow = start
    start.next = head
    for i in range(n):
        fast = fast.next
    if not fast:
        return None
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next

    return start.next

# 21.合并两个有序链表
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    head = ListNode(0)
    pointer = head
    while l1 and l2:
        if l1.val < l2.val:
            pointer.next = ListNode(l1.val)
            l1 = l1.next
        else:
            pointer.next = ListNode(l2.val)
            l2 = l2.next
        pointer = pointer.next
    if l1:
        pointer.next = l1
    if l2:
        pointer.next = l2
    return head.next

# 23. 合并k个排序链表
def mergeKLists(lists) -> ListNode:
    if not lists:
        return None
    index = 1
    while index < len(lists):
        for i in range(0, len(lists) - index, index * 2):
            lists[i] = mergeTwoLists(lists[i], lists[i + index])
        index *= 2
    return lists[0]

# 24. 两两交换链表中的节点
def swapPairs(head: ListNode) -> ListNode:
    p_head = tmp = ListNode(0)
    p_head.next = head
    while tmp.next and tmp.next.next:
        start, end = tmp.next, tmp.next.next
        tmp.next = end
        start.next = end.next
        end.next = start
        tmp = start
    return p_head.next

# 25. k个一组翻转链表
def reverseKGroup(head: ListNode, k: int):
    p_head = curr = ListNode(0)
    while True:
        count = k
        stack = []
        tmp = head
        while count and tmp:
            stack.append(tmp)
            tmp = tmp.next
            count -= 1
        if count:  # 栈中个数小于k
            curr.next = head
            break
        while stack:
            curr.next = stack.pop()
            curr = curr.next
        curr.next = tmp
        head = tmp
    return p_head.next


# 141. 环形链表
def hasCycle(head):
    """
    :type head: ListNode
    :rtype: bool
    """
    if not head:
        return False
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# 142. 环形链表II
def detectCycle(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return None
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    if slow != fast or fast.next is None:
        return None
    fast = head
    while fast != slow:
        fast, slow = fast.next, slow.next
    return fast

head = ListNode(1)
#head.next = ListNode(2)
#head.next.next = ListNode(3)
#head.next.next.next = head
print(detectCycle(head))

# 306. 反转链表
def reverseList(head: ListNode) -> ListNode:
    p_head = ListNode(0)
    p_prev = None
    p_node = head
    while p_node:
        p_next = p_node.next
        if not p_next:
            p_head.next = p_node
        p_node.next = p_prev
        p_prev = p_node
        p_node = p_next
    return p_head.next