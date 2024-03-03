class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# 206. 反转链表
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        while head:
            next = head.next
            head.next = prev
            prev = head
            head = next
        return prev

# 25. K个一组翻转链表
class Solution(object):
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        p_head = curr = ListNode(0)
        while True:
            count = k
            stack = []
            tmp = head
            while count and tmp:
                stack.append(tmp)
                tmp = tmp.next
                count -= 1
            if count: # 栈中个数小于k
                curr.next = head
                break
            while stack:
                curr.next = stack.pop()
                curr = curr.next
            curr.next = tmp
            head = tmp
        return p_head.next

# 21.合并2个有序链表
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        head = ListNode(0)
        res = head
        while list1 and list2:
            if list1.val < list2.val:
                res.next = ListNode(list1.val)
                res = res.next
                list1 = list1.next
            else:
                res.next = ListNode(list2.val)
                res = res.next
                list2 = list2.next
        if list1:
            res.next = list1
        if list2:
            res.next = list2
        return head.next

# 92. 反转链表II
class Solution(object):
    def reverseBetween(self, head, m, n):
        p_head = ListNode(0)
        p_head.next = head
        pre = p_head
        for i in range(1, m):
            pre = pre.next
        head = pre.next
        for i in range(m, n):
            nxt = head.next
            head.next = nxt.next
            nxt.next = pre.next
            pre.next = nxt

        return p_head.next

# 23. 合并K个升序链表
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        head = point = ListNode(0)
        while list1 and list2:
            if list1.val < list2.val:
                point.next = list1
                list1 = list1.next
            else:
                point.next = list2
                list2 = list2.next
            point = point.next
        if list1:
            point.next = list1
        else:
            point.next = list2
        return head.next

    def merge(self, lists, left, right):
        if left == right:
            return lists[left]
        mid = left + (right - left) // 2
        l1 = self.merge(lists, left, mid)
        l2 = self.merge(lists, mid+1, right)
        return self.mergeTwoLists(l1,l2)

    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        if not lists:
            return None
        return self.merge(lists, 0, len(lists)-1)

# 143. 重排链表
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """

        def findMid(head):
            slow, fast = head, head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            return slow

        def reverseList(head):
            prev = None
            while head:
                next = head.next
                head.next = prev
                prev = head
                head = next
            return prev

        def mergeList(l1, l2):
            while l1 and l2:
                l1_tmp = l1.next
                l2_tmp = l2.next
                l1.next = l2
                l1 = l1_tmp
                l2.next = l1
                l2 = l2_tmp

        mid = findMid(head)
        l1 = head
        l2 = mid.next
        mid.next = None
        l2 = reverseList(l2)
        mergeList(l1, l2)

# 142. 环形链表II
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                fast = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        return None

# 148.排序链表
class Solution(object):
    def sortList(self, head):
        return self.merge_sort(head)

    def merge_sort(self, head):
        if not head or not head.next:
            return head
        mid = self.get_mid(head)
        right = self.merge_sort(mid.next)
        mid.next = None
        left = self.merge_sort(head)
        return self.merge_two_lists(left, right)

    def get_mid(self, head):
        if not head:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
    def merge_two_lists(self, l1, l2):
        dummy = ListNode(0)
        node =dummy

        while l1 or l2:
            val1 = 2 **31 - 1 if not l1 else l1.val
            val2 = 2 ** 31 - 1 if not l2 else l2.val
            if val1 < val2:
                node.next = l1
                l1 = l1.next
            else:
                node.next = l2
                l2 = l2.next
            node = node.next
        return dummy.next

# 24.两两交换链表中的节点
class Solution(object):
    def swapPairs(self, head):
        if not head:
            return None
        p_head = tmp = ListNode(0)
        tmp.next = head
        while tmp.next and tmp.next.next:
            p1 = tmp.next
            p2 = tmp.next.next
            tmp.next = p2
            p1.next = p2.next
            p2.next = p1
            tmp = p1
        return p_head.next