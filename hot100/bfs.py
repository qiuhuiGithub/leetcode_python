# 301. 删除无效括号
class Solution(object):
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """

        def isValid(s):
            count = 0
            for c in s:
                if c == '(':
                    count += 1
                elif c == ')':
                    count -= 1
                    if count < 0:
                        return False
            return count == 0

        ans = []
        curSet = set([s])
        while True:
            for cur in curSet:
                if isValid(cur):
                    ans.append(cur)
            if len(ans) > 0:
                return ans
            tmp = set()
            for cur in curSet:
                for i in range(len(cur)):
                    if i > 0 and cur[i] == cur[i - 1]:
                        continue
                    if cur[i] in ['(', ')']:
                        tmp.add(cur[:i] + cur[i + 1:])
            curSet = tmp
        return ans


# 1293. 网格中的最短路径
class Solution(object):
    def shortestPath(self, grid, k):
        if not grid or grid == [[0]]:
            return 0
        m, n = len(grid), len(grid[0])
        k = min(k, m + n - 3)
        queue = [(0, 0, k)]
        visits = set([(0, 0, k)])
        step = 0
        while len(queue) > 0:
            step += 1
            cnt = len(queue)
            for _ in range(cnt):
                dx, dy, rest = queue.pop(0)
                for dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dx1, dy1 = dx + dir[0], dy + dir[1]
                    if 0 <= dx1 < m and 0 <= dy1 < n:
                        if grid[dx1][dy1] == 0 and (dx1, dy1, rest) not in visits:
                            if dx1 == m - 1 and dy1 == n - 1:
                                return step
                            queue.append((dx1, dy1, rest))
                            visits.add((dx1, dy1, rest))
                        elif grid[dx1][dy1] == 1 and rest > 0 and (dx1, dy1, rest - 1) not in visits:
                            queue.append((dx1, dy1, rest - 1))
                            visits.add((dx1, dy1, rest - 1))
        return -1

# 207.课程表
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        edges = collections.defaultdict(list)
        indeg = [0] * numCourses
        for info in prerequisites:
            edges[info[1]].append(info[0])
            indeg[info[0]] += 1
        q = collections.deque([u for u in range(numCourses) if indeg[u] ==0])
        visited = 0
        while q:
            visited += 1
            u = q.popleft()
            for v in edges[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return visited == numCourses

# 79. 单词搜索
class Solution(object):
    def exist(self, board, word):
        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]

        def backtrack(i, j, k):
            if k == len(word):
                return True
            if i < 0 or i >= m or j < 0 or j >= n or visited[i][j] or board[i][j] != word[k]:
                return False
            visited[i][j] = True
            res = backtrack(i + 1, j, k + 1) or backtrack(i - 1, j, k + 1) or backtrack(i, j - 1, k + 1) or backtrack(i, j + 1, k + 1)
            visited[i][j] = False
            return res

        for i in range(m):
            for j in range(n):
                if backtrack(i, j, 0):
                    return True
        return False