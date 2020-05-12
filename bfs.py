# encoding=utf-8
# 127.单词接龙
def ladderLength(beginWord, endWord, wordList):
    """
    :type beginWord: str
    :type endWord: str
    :type wordList: List[str]
    :rtype: int
    """
    # 单向BFS，过不了
    # q = [beginWord]
    # res = 0
    # while q:
    #     res += 1
    #     for sz in range(len(q)):
    #         hop = q.pop(0)
    #         if hop == endWord:
    #             return res
    #         for idx in range(len(wordList)):
    #             if not wordList[idx]:
    #                 continue
    #             diff = 0
    #             for i in range(len(wordList[idx])):
    #                 if wordList[idx][i] != hop[i]:
    #                     diff += 1
    #                 if diff > 1:
    #                     break
    #             if diff <= 1:
    #                 q.append(wordList[idx])
    #                 wordList[idx] = ""
    # return 0

    # 双向BFS
    if endWord not in wordList:
        return 0
    wordSet = set(wordList)
    head, tail = {beginWord}, {endWord}
    tmp = list('abcdefghijklmnopqrstuvwxyz')
    res = 1
    while head:
        if len(head) > len(tail):
            head, tail = tail, head
        q = set()
        for cur in head:
            for i in range(len(cur)):
                for j in tmp:
                    word = cur[:i] + j + cur[i + 1:]
                    if word in tail:
                        return res + 1
                    if word in wordSet:
                        q.add(word)
                        wordSet.remove(word)
        head = q
        res += 1
    return 0

# 130.被围绕的区域
def solve(board):
    """
    :type board: List[List[str]]
    :rtype: None Do not return anything, modify board in-place instead.
    """
    if not board:
        return
    row, col = len(board), len(board[0])

    def bfs(i, j):
        queue = [(i, j)]
        while queue:
            (t_i, t_j) = queue.pop(0)
            if 0 <= t_i < len(board) and 0 <= t_j < len(board[0]) and board[t_i][t_j] == 'O':
                board[t_i][t_j] = 'B'
                for (r, c) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((t_i + r, t_j + c))

    def dfs(i, j):
        board[i][j] = 'B'
        for (r, c) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            t_i, t_j = i + r, j + c
            if 0 <= t_i < len(board) and 0 <= t_j < len(board[0]) and board[t_i][t_j] == 'O':
                dfs(t_i, t_j)

    for i in range(row):
        if board[i][0] == 'O':
            bfs(i, 0)
        if board[i][col - 1] == 'O':
            bfs(i, col - 1)
    for j in range(col):
        if board[0][j] == 'O':
            bfs(0, j)
        if board[row - 1][j] == 'O':
            bfs(row - 1, j)
    for i in range(row):
        for j in range(col):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            if board[i][j] == 'B':
                board[i][j] = 'O'