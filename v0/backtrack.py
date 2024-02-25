# encoding=utf-8
# 17. 电话号码字母组合
def letterCombinations(digits):
    res = []
    phone = {"0": " ", "1": "*", "2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv",
             "9": "wxyz"}
    if not digits:
        return res

    def backtrack(combine, digits):
        if not digits:
            res.append(combine)
        else:
            letters = phone[digits[0]]
            for letter in letters:
                backtrack(combine + letter, digits[1:])

    backtrack("", digits)
    return res


# 22. 括号生成
def generateParenthesis(n):
    res = []
    if n < 1:
        return res

    def backtrack(curr, open, close):
        if len(curr) == 2 * n:
            res.append(curr)
        if open < n:
            backtrack(curr + '(', open + 1, close)
        if close < open:
            backtrack(curr + ')', open, close + 1)

    backtrack("", 0, 0)
    return res


# 36.有效的数独
def isValidSudoku(board):
    rows = [[False for _ in range(9)] for _ in range(9)]
    columns = [[False for _ in range(9)] for _ in range(9)]
    boxs = [[False for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            if board[i][j] != '.':
                box_index = (i // 3) * 3 + j // 3
                num = int(board[i][j]) - 1
                if rows[i][num] or columns[j][num] or boxs[box_index][num]:
                    return False
                else:
                    rows[i][num], columns[j][num], boxs[box_index][num] = True, True, True
    return True


# 37.解数独
def solveSudoku(board):
    def backtrack(board, i, j):
        while board[i][j] != '.':
            if j == 8:
                i += 1
                j = 0
            else:
                j += 1
            if i >= 9:
                return True
        for num in range(9):
            block_index = (i // 3) * 3 + j // 3
            if not row[i][num] and not col[j][num] and not block[block_index][num]:
                board[i][j] = str(num + 1)
                row[i][num] = True
                col[j][num] = True
                block[block_index][num] = True
                if backtrack(board, i, j):
                    return True
                else:
                    board[i][j] = '.'
                    row[i][num] = False
                    col[j][num] = False
                    block[block_index][num] = False
        return False

    row = [[False for _ in range(9)] for _ in range(9)]
    col = [[False for _ in range(9)] for _ in range(9)]
    block = [[False for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            if board[i][j] != '.':
                block_index = (i // 3 * 3) + (j // 3)
                num = int(board[i][j]) - 1
                row[i][num] = True
                col[j][num] = True
                block[block_index][num] = True
    backtrack(board, 0, 0)


# 39.组合总和
def combinationSum(candidates, target):
    res = []

    def backtrack(start, path, target):
        if target < 0:
            return
        if target == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, target - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return res


# print(combinationSum([2, 3], 8))


# 40.组合总和II

def combinationSum2(candidates, target):
    res = []
    candidates.sort()

    def backtrack(start, path, target):
        if target < 0:
            return
        if target == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            path.append(candidates[i])
            backtrack(i + 1, path, target - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return res


# 46.全排列
def permute(nums):
    res = []

    def backtrack(path, nums):
        if len(path) == len(nums):
            res.append(path[:])
            return

        for num in nums:
            if num in path:
                continue
            path.append(num)
            backtrack(path, nums)
            path.pop()

    backtrack([], nums)
    return res


# print(permute([1, 2, 3]))


# 47.全排列II
def permuteUnique(nums):
    res = []

    def backtrack(path, nums):
        if len(nums) == 0:
            tmp = path[:]
            if tmp not in res:
                res.append(tmp)
        for i in range(len(nums)):
            path.append(nums[i])
            backtrack(path, nums[0:i] + nums[i + 1:])
            path.pop()

    backtrack([], nums)
    return res


# 51.N皇后
def solveNQueens(n):
    def isValidNQueens(board, row, col):
        # 检查列是否合法
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # 检查右对角线是否合法
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i, j = i - 1, j + 1

        # 检查左对角线适合合法
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i, j = i - 1, j - 1

        return True

    def backtrack(board, row):
        if row == len(board):
            tmp = []
            for item in board:
                tmp.append(''.join(item))
            res.append(tmp)
        for col in range(len(board)):
            if not isValidNQueens(board, row, col):
                continue
            board[row][col] = 'Q'
            backtrack(board, row + 1)
            board[row][col] = '.'

    res = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(board, 0)
    return res


# print(solveNQueens(4))


# 52.N皇后II
def totalNQueens(n):
    def isValidNQueens(board, row, col):
        # 检查列是否合法
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # 检查右对角线是否合法
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i, j = i - 1, j + 1

        # 检查左对角线适合合法
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i, j = i - 1, j - 1

        return True

    def backtrack(board, row):
        if row == len(board):
            tmp = []
            for item in board:
                tmp.append(''.join(item))
            res.append(tmp)
        for col in range(len(board)):
            if not isValidNQueens(board, row, col):
                continue
            board[row][col] = 'Q'
            backtrack(board, row + 1)
            board[row][col] = '.'

    res = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(board, 0)
    return len(res)


# 60.第K个排列
def getPermutation(n, k):
    """
    :type n: int
    :type k: int
    :rtype: str
    """
    res = []
    candidate = []
    factorials = [1 for _ in range(n + 1)]
    fact = 1
    for i in range(1, n + 1):
        candidate.append(i)
        fact *= i
        factorials[i] = fact
    k -= 1
    for i in range(n - 1, -1, -1):
        index = k // factorials[i]
        res.append(candidate.pop(index))
        k -= index * factorials[i]
    return ''.join([str(i) for i in res])


# print(getPermutation(3, 3))


# 77.组合总和
def combine(n, k):
    """
    :type n: int
    :type k: int
    :rtype: List[List[int]]
    """
    res = []

    def backtrack(path, nums):
        if len(path) == k:
            res.append(path[:])
            return
        for i in range(len(nums)):
            path.append(nums[i])
            backtrack(path, nums[i + 1:])
            path.pop()

    backtrack([], [i + 1 for i in range(n)])
    return res


# print(combine(4, 2))


# 78.子集
def subsets(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    res = []

    def backtrack(path, nums):
        if len(nums) >= 0:
            res.append(path[:])
        for i in range(len(nums)):
            path.append(nums[i])
            backtrack(path, nums[i + 1:])
            path.pop()

    backtrack([], nums)
    return res


# print(subsets([1, 2, 3]))


# 79. 单词搜索
def exist(board, word):
    """
    :type board: List[List[str]]
    :type word: str
    :rtype: bool
    """

    def backtrack(board, word, i, j, k):
        if k >= len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
            return False
        tmp = board[i][j]
        board[i][j] = 0
        res = backtrack(board, word, i - 1, j, k + 1) or backtrack(board, word, i + 1, j, k + 1) or \
              backtrack(board, word, i, j - 1, k + 1) or backtrack(board, word, i, j + 1, k + 1)
        board[i][j] = tmp
        return res

    if not board or not board[0]:
        return False
    m, n = len(board), len(board[0])
    for i in range(m):
        for j in range(n):
            if backtrack(board, word, i, j, 0):
                return True
    return False


# print(exist([['A', 'B', 'C', 'E'],
#              ['S', 'F', 'C', 'S'],
#              ['A', 'D', 'E', 'E']], 'ABCCED'))


# 89. 格雷编码
def grayCode(n):
    """
    :type n: int
    :rtype: List[int]
    """
    res = []
    visit = [False for _ in range(2 ** n)]

    def backtrack(curr):
        if len(res) == 2 ** n:
            return True
        res.append(curr)
        visit[curr] = True
        for i in range(n):
            next = curr ^ (1 << i)
            if visit[next]:
                continue
            backtrack(next)
        visit[curr] = False
        return False

    backtrack(0)
    return res


# print(grayCode(2))

# 90.子集
def subsetsWithDup(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    res = []
    nums.sort()

    def backtrack(idx, path, nums):
        if idx <= len(nums):
            res.append(path[:])
        for i in range(idx, len(nums)):
            if i > idx and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path, nums)
            path.pop()

    backtrack(0, [], nums)
    return res


print(subsetsWithDup([1, 2, 2]))


# 113. 路径总和
def pathSum(root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: List[List[int]]
    """
    res = []
    if not root:
        return res

    def backtrack(root, path, sum):
        if not root:
            return
        sum -= root.val
        path.append(root.val)
        if not root.left and not root.right and sum == 0:
            res.append(path[:])
        backtrack(root.left, path[:], sum)
        backtrack(root.right, path[:], sum)

    backtrack(root, [], sum)
    return res
