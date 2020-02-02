# encoding=utf-8

"""
二叉树遍历
"""


class BiTree:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def create_tree(lst):
    if not lst:
        return None
    queue = []
    root = BiTree(lst.pop(0))
    queue.append(root)
    while queue and lst:
        node = queue.pop(0)
        left_val = lst.pop(0)
        if left_val == 0:
            node.left = None
        else:
            node.left = BiTree(left_val)
            queue.append(node.left)

        right_val = lst.pop(0)
        if right_val == 0:
            node.right = None
        else:
            node.right = BiTree(right_val)
            queue.append(node.right)
    return root


"""
递归
"""


def pre_order(root):
    if not root:
        return
    print(root.val)
    if root.left:
        pre_order(root.left)
    if root.right:
        pre_order(root.right)


def in_order(root):
    if not root:
        return
    if root.left:
        pre_order(root.left)
    print(root.val)
    if root.right:
        pre_order(root.right)


def post_order(root):
    if not root:
        return
    if root.left:
        pre_order(root.left)
    if root.right:
        pre_order(root.right)
    print(root.val)


def level_order(root):
    queue = []
    queue.append(root)
    while queue:
        node = queue.pop(0)
        print(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)


"""
非递归
"""


def pre_order_1(root):
    if not root:
        return
    stack = []
    node = root
    while node or stack:
        while node:
            print(node.val)
            stack.append(node)
            node = node.left
        node = stack.pop()
        node = node.right


def in_order_1(root):
    if not root:
        return
    stack = []
    node = root
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        print(node.val)
        node = node.right


def post_order_1(root):
    if not root:
        return
    stack1 = []
    stack2 = []
    stack1.append(root)
    while stack1:
        node = stack1.pop()
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)
        stack2.append(node)
    while stack2:
        node = stack2.pop()
        print(node.val)


if __name__ == '__main__':
    root = create_tree([1, 2, 3, 4, 5])
    # 递归
    pre_order(root)
    in_order(root)
    post_order(root)

    # 非递归
    pre_order_1(root)
    in_order_1(root)
    post_order_1(root)

    # 层序遍历
    level_order(root)
