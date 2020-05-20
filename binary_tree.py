# encoding=utf-8

"""
二叉树遍历
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def create_tree(lst):
    if not lst:
        return None
    queue = []
    root = TreeNode(lst.pop(0))
    queue.append(root)
    while queue and lst:
        node = queue.pop(0)
        left_val = lst.pop(0)
        if left_val == 0:
            node.left = None
        else:
            node.left = TreeNode(left_val)
            queue.append(node.left)

        right_val = lst.pop(0)
        if right_val == 0:
            node.right = None
        else:
            node.right = TreeNode(right_val)
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


# 面试题68. 二叉树的最近公共祖先
def lowestCommonAncestor(root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    if not root:
        return None
    if p == root or q == root:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    if left:
        return left
    if right:
        return right
    return None

# 98. 验证二叉搜索树
def isValidBST(root):
    """
    :type root: TreeNode
    :rtype: bool
    """

    def valid(node, min, max):
        if not node:
            return True
        if node.val <= min or node.val >= max:
            return False
        return valid(node.left, min, node.val) and valid(node.right, node.val, max)

    return valid(root, -2 ** 31 - 1, 2 ** 31)

# 100. 相同的树
def isSameTree(p, q):
    """
    :type p: TreeNode
    :type q: TreeNode
    :rtype: bool
    """
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val == q.val:
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)

# 101. 对称二叉树
def isSymmetric(root):
    """
    :type root: TreeNode
    :rtype: bool
    """

    def isMirror(left, right):
        if not left and not right:
            return True
        if left and right and left.val == right.val:
            return isMirror(left.left, right.right) and isMirror(left.right, right.left)
        return False

    return isMirror(root, root)

# 102. 二叉树的层序遍历
def levelOrder(root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    res = []
    if not root:
        return []
    queue = []
    queue.append(root)
    while queue:
        level = []
        for i in range(len(queue)):
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level)
    return res

# 103. 二叉树的锯齿形遍历
def zigzagLevelOrder(root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    res = []
    if not root:
        return []
    queue = []
    queue.append(root)
    flag = False
    while queue:
        level = []
        flag = not flag
        for i in range(len(queue)):
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level if flag else level[::-1])
    return res

# 104. 二叉树的最大深度
def maxDepth(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
        return 0
    return max(maxDepth(root.left), maxDepth(root.right)) + 1

# 105. 从前序与中序遍历序列构造二叉树
def buildTree(preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """
    if not inorder:
        return
    mid = inorder.index(preorder[0])
    root = TreeNode(preorder[0])
    root.left = buildTree(preorder[1:mid + 1], inorder[:mid])
    root.right = buildTree(preorder[mid + 1:], inorder[mid + 1:])

# 106. 从中序与后序遍历序列构造二叉树
def buildTree_1(inorder, postorder):
    """
    :type inorder: List[int]
    :type postorder: List[int]
    :rtype: TreeNode
    """
    if not inorder:
        return
    length = len(postorder)
    root = TreeNode(postorder[length - 1])
    mid = inorder.index(postorder[length - 1])
    root.left = buildTree_1(inorder[:mid], postorder[:mid])
    root.right = buildTree_1(inorder[mid + 1:], postorder[mid:length - 1])

# 107. 二叉树的层序遍历 II
def levelOrderBottom(root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    res = []
    if not root:
        return res
    queue = []
    queue.append(root)
    while queue:
        level = []
        for i in range(len(queue)):
            node = queue.pop(0)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            level.append(node.val)
        res.append(level)
    return res[::-1]

# 111. 二叉树的最小深度
def minDepth(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
        return 0
    left, right = minDepth(root.left), minDepth(root.right)
    return min(left, right) + 1 if left and right else 1 + left + right
    # if not root:
    #     return 0
    # queue = [root]
    # depth = 1
    # while queue:
    #     for i in range(len(queue)):
    #         node = queue.pop(0)
    #         if not node.left and not node.right:
    #             return depth
    #         if node.left:
    #             queue.append(node.left)
    #         if node.right:
    #             queue.append(node.right)
    #     depth += 1

# 112. 路径总和
def hasPathSum(root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: bool
    """
    if not root:
        return False
    if not root.left and not root.right:
        return sum - root.val == 0
    return hasPathSum(root.left, sum - root.val) or hasPathSum(root.right, sum - root.val)



# 199. 二叉树的右视图
def rightSideView(root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    res = []
    if not root:
        return res
    queue = [root]
    while queue:
        sz = len(queue)
        for i in range(sz):
            node = queue.pop(0)
            if i == sz - 1:
                res.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return res

# 543. 二叉树的直径
class Solution(object):
    def diameterOfBinaryTree(self, root):
        self.ans = 1
        def depth(node):
            if not node:
                return 0
            L = depth(node.left)
            R = depth(node.right)
            self.ans = max(self.ans, L+R+1)
            return max(L,R) + 1
        depth(root)
        return self.ans-1