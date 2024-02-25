# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# 102. 二叉树的层序遍历
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res = []
        if not root:
            return res
        queue = [root]
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


# 236. 二叉树的最近公共祖先
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left:
            return right
        if not right:
            return left
        return root


# 110. 平衡二叉树
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        self.res = True

        def get_depth(root):
            if not root:
                return 0
            left = get_depth(root.left) + 1
            right = get_depth(root.right) + 1
            if abs(left - right) > 1:
                self.res = False
            return max(left, right)

        get_depth(root)
        return self.res


# 226. 翻转二叉树
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        root.left = right
        root.right = left
        return root


# 101.对称二叉树
class Solution(object):
    def isSymmetric(self, root):
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

# 98. 验证二叉搜索树
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        def valid(root, min, max):
            if not root:
                return True
            if root.val <= min or root.val >= max:
                return False
            return valid(root.left, min, root.val) and valid(root.right, root.val, max)

        return valid(root, -float('inf'), float('inf'))

# 543. 二叉树的直径
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.diameter = 0

        def get_depth(root):
            if not root:
                return 0
            left = get_depth(root.left)
            right = get_depth(root.right)
            self.diameter = max(self.diameter, left + right + 1)
            return max(left, right) + 1

        get_depth(root)
        return self.diameter - 1

# 662. 二叉树的最大宽度
class Solution(object):
    def widthOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        idx = 1
        width = 1
        queue = [(root, idx)]
        while queue:
            width = max(width, queue[-1][1] - queue[0][1] + 1)
            for i in range(len(queue)):
                root, idx = queue.pop(0)
                if root.left:
                    queue.append([root.left, idx * 2])
                if root.right:
                    queue.append([root.right, idx * 2 + 1])
        return width

# 104. 二叉树的最大深度
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def get_depth(root):
            if not root:
                return 0
            return 1 + max(get_depth(root.left), get_depth(root.right))

        return get_depth(root)

# 111. 二叉树的最小深度
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        min_depth = float('inf')
        if root.left:
            min_depth = min(min_depth, self.minDepth(root.left))
        if root.right:
            min_depth = min(min_depth, self.minDepth(root.right))
        return min_depth + 1

# 124. 二叉树的最大路径和
class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.res = float('-inf')

        def max_sum(root):
            if not root:
                return 0
            max_left = max(max_sum(root.left), 0)
            max_right = max(max_sum(root.right), 0)
            path_val = root.val + max_left + max_right
            self.res = max(self.res, path_val)

            return root.val + max(max_left, max_right)

        max_sum(root)
        return self.res

# 105. 从前序与中序遍历构造二叉树
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not inorder:
            return
        mid = inorder.index(preorder[0])
        root = TreeNode(preorder[0])
        root.left = self.buildTree(preorder[1:mid+1], inorder[:mid])
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])
        return root

# 106. 从中序与后序遍历构造二叉树
class Solution(object):
    def buildTree(self, inorder, postorder):
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
        root.left = self.buildTree(inorder[:mid], postorder[:mid])
        root.right = self.buildTree(inorder[mid + 1:], postorder[mid:length - 1])
        return root

# 112. 路径总和
class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: bool
        """
        self.res = False

        def backtrack(root, targetSum):
            if not root:
                return
            targetSum -= root.val
            if not root.left and not root.right and targetSum == 0:
                self.res = True
            backtrack(root.left, targetSum)
            backtrack(root.right, targetSum)

        backtrack(root, targetSum)
        return self.res

# 113. 路径总和II
class Solution(object):
    def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        self.res = []
        if not root:
            return self.res

        def backtrack(root, targetSum, path):
            if not root:
                return
            path.append(root.val)
            targetSum -= root.val
            if not root.left and not root.right and targetSum == 0:
                self.res.append(path[:])
            backtrack(root.left, targetSum, path[:])
            backtrack(root.right, targetSum, path[:])

        backtrack(root, targetSum, [])
        return self.res

# 129. 根节点到叶子结点数字之和
class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.res = []
        if not root:
            return self.res

        def backtrack(root, path):
            if not root:
                return 0
            path = path + str(root.val)
            if not root.left and not root.right:
                self.res.append(path)
            backtrack(root.left, path)
            backtrack(root.right, path)

        backtrack(root, "")
        return sum([int(i) for i in self.res])