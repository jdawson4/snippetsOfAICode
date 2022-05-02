# Author: Jacob Dawson
#
# Just a small bit of code I made to solve a (actually very easy) programming
# challenge from Hackerrank.
# This solution uses BFS to solve the problem, when honestly a simpler
# solutution would do just as well. I just like GSA :)

def printer(grid):
    # useful for visualizing
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            print(grid[i][j], end='')
        print('')

def genNewGrid(grid, newLoc):
    # a helper for my helper!
    # it creates a new grid from the current grid. However,
    # it will remove the m from loc and add it to newLoc
    newGrid = []
    for i in range(len(grid)):
        row = []
        for j in range(len(grid[i])):
            if(i==newLoc[0] and j==newLoc[1]):
                # add the m!
                row.append('b')
            else:
                # check if it's the princess there:
                if grid[i][j]=='d':
                    # then retain the princess.
                    row.append('d')
                else:
                    # if not, add nothing!
                    row.append('-')
        newGrid.append(row)
    return newGrid


def generateSuccessors(grid):
    # given a state (the grid), return all possible successor (m moves by 1)
    # we are also going to return the move (ex: 'DOWN') which gets us there,
    # if that's a valid move.

    # there should normally be 4 successors, in the cardinal directions
    moves = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    #print('here5')

    # let's find our current position
    location = (0,0)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if(grid[i][j] == 'b'):
                # location found!
                location = (i,j)

    # now we determine which ones are valid

    # let's just manually compute the four cases:
    successors = []
    for move in moves:
        if move=='UP':
            newLoc = (location[0]-1, location[1])
            if((location[0]-1) >= 0):
                newGrid = genNewGrid(grid, newLoc)
                successorTuple = (move, newGrid)
                successors.append(successorTuple)
        elif move=='RIGHT':
            newLoc = (location[0], location[1]+1)
            if((newLoc[1]) <= len(grid[0])):
                newGrid = genNewGrid(grid, newLoc)
                successorTuple = (move, newGrid)
                successors.append(successorTuple)
        elif move=='DOWN':
            newLoc = (location[0]+1, location[1])
            if((newLoc[0]) <= len(grid)):
                newGrid = genNewGrid(grid, newLoc)
                successorTuple = (move, newGrid)
                successors.append(successorTuple)
        elif move=='LEFT':
            newLoc = (location[0], location[1]-1)
            if((newLoc[1]) >= 0):
                newGrid = genNewGrid(grid, newLoc)
                successorTuple = (move, newGrid)
                successors.append(successorTuple)
    return successors

def isGoal(grid):
    # given a state (a grid), return a bool (whether or not m is on top of p)
    # this is equivalent to the question "has p disappeared"
    for row in grid:
        for item in row:
            if (item=='d'):
                return False # then this is NOT the end goal
            # /\ generally bad practice but it's actually more efficient
            # to it do this way because it limits iteration
    return True # because we've iterated through anything and haven't found p!

def next_move(n,grid):
    #print all the moves here

    # okay, so this "search" issue is actually a pretty well solved problem.
    # I'm going to use a solution called breadth first search (BFS),
    # because I've never done that before and it looks cool

    # we're gonna implement BFS via GSA (Generic Search Algorithm)
    # because it's famous and well documented
    closed = []
    openStack = []
    # note: we treat this as a queue! appends and pops from 0 only!

    # this code does BFS
    # we need the notion of nodes. We will use these to determine path
    current = dict()
    current['state']=grid
    current['parent'] = None
    current['action'] = None

    #print('here0')
    while not(isGoal(current['state'])):
        #print('here1')
        closed.append(current['state'])
        successors = generateSuccessors(current['state'])
        for m, s in successors:
            #print('here2')
            if not s in closed:
                newNode = dict()
                newNode['state'] = s
                newNode['parent'] = current
                newNode['action'] = m
                openStack.append(newNode)
        current = openStack.pop(0)
        # fun fact: removing this 0 turn this into DFS!
    path = list()
    while(current['parent'] != None):
        #print('here3')
        path.append(current['action'])
        current = current['parent']

    return path[0]

def generateRandomGrid(n):
    # generates a random grid of nxn size and places the player and princess
    # randomly if a certain epsilon is crossed
    coin = ['p','m']
    epsilon = 0.05
    placedPlayer = False
    placedPrincess = False
    grid = []
    for i in range(n):
        row = list()
        for j in range(n):
            if (random.random()<epsilon):
                choice = random.choice(coin)
                if((choice=='p') and (not placedPrincess)):
                    placedPrincess = True
                    row.append('p')
                elif((choice=='m') and (not placedPlayer)):
                    placedPlayer = True
                    row.append('m')
                else:
                    row.append('-')
            else:
                row.append('-')
        grid.append(row)
    if not placedPlayer:
        # by default, we stick him in position 0,0
        grid[0][0] = 'm'
    if not placedPrincess:
        grid[n-1][n-1] = 'p'
    printer(grid)
    return grid
# Tail starts here

if __name__ == "__main__":
    pos = [int(i) for i in input().strip().split()]
    board = [[j for j in input().strip()] for i in range(5)]
    next_move(len(board), board)
