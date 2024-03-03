# 200. 岛屿数量
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        m, n = len(grid), len(grid[0])
        res = 0

        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
                return 0
            grid[i][j] = '0'
            for dir in dirs:
                dfs(i + dir[0], j + dir[1])

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    res += 1
                    dfs(i, j)
        return res

# 695. 岛屿的最大面积
class Solution(object):
    def maxAreaOfIsland(self, grid):
        if not grid:
            return 0
        max_area = 0
        m, n = len(grid), len(grid[0])

        def dfs(i, j):
            if i >= m or i < 0 or j >= n or j < 0 or grid[i][j] == 0:
                return 0
            grid[i][j] = 0
            area = 1
            for dir in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                i1, j1 = i + dir[0], j + dir[1]
                area += dfs(i1, j1)
            return area

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    area = dfs(i, j)
                    max_area = max(max_area, area)
        return max_area