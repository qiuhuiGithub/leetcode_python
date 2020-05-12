# encoding=utf-8
# 200.岛屿数量
def numIslands(grid):
    """
    :type grid: List[List[str]]
    :rtype: int
    """

    def dfs(grid, row, col):
        grid[row][col] = '0'
        if row - 1 >= 0 and grid[row - 1][col] == '1':
            dfs(grid, row - 1, col)
        if row + 1 < len(grid) and grid[row + 1][col] == '1':
            dfs(grid, row + 1, col)
        if col - 1 >= 0 and grid[row][col - 1] == '1':
            dfs(grid, row, col - 1)
        if col + 1 < len(grid[0]) and grid[row][col + 1] == '1':
            dfs(grid, row, col + 1)

    if not grid:
        return 0
    length, width = len(grid), len(grid[0])
    num = 0
    for i in range(length):
        for j in range(width):
            if grid[i][j] == '1':
                num += 1
                dfs(grid, i, j)
    return num