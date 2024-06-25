import numpy as np # This is only used for creating empty sparseTable, empty precomputedTable and finding log 2 based

class FullPreProcessing:

    def __init__(self):
        self.precomputedTable = None # store precomputed table to use in future

    def full_preprocessing(self, MAQInput, corners): # it returns precomputed table
        # Initializing variables
        startCorner, endCorner = corners
        current_row_index, current_col_index = startCorner
        end_row_index, end_col_index = endCorner
        numOfCol = len(MAQInput[0])
        last_col_index = numOfCol - 1
        numOfElement = ((end_row_index - current_row_index + 1) * numOfCol) - current_col_index - (numOfCol - end_col_index - 1)
        index = 0

        # r = row number of MAQInput
        # c = column number of MAQInput
        # numOfElement = r * c
        # Initializing empty 2-D array (numOfElement x numOfElement) called precomputedTable with default as 0
        precomputedTable = np.full((numOfElement, numOfElement), None)

        # for loop fills same row and same column of preceomputedTable with MAQInput elements
        for i in range(numOfElement):
            precomputedTable[index][index] = MAQInput[current_row_index][current_col_index]
            index += 1
            if current_col_index == last_col_index:
                current_col_index = 0
                current_row_index += 1
                continue
            current_col_index += 1

        current_row = 0
        current_col = 0

        # Table is ready to precompute
        precomputedTable = self.precompute(precomputedTable, current_row, current_col)
        self.precomputedTable = precomputedTable

        return precomputedTable


    def precompute(self, precomputedTable, row, col): # it precomputes other cells of precomptedTable recursively
        # Initializing varibles
        indexOfLastCol = len(precomputedTable) - 1
        temp_row = row
        temp_col = col

        # while temp column is not equal last column of precompted table, find other cell
        while temp_col != indexOfLastCol:
            # compare lower cross two elements and select min, then fill upper cross element of them with min value
            precomputedTable[temp_row][temp_col + 1] = min(precomputedTable[temp_row][temp_col], precomputedTable[temp_row + 1][temp_col + 1])
            temp_row += 1
            temp_col += 1

        # if current column reach the last column of precomputed table, that means precomputing process finished.
        if col == indexOfLastCol:
            return precomputedTable

        # Otherwise, continue precomputing process by increasing current column by one
        return self.precompute(precomputedTable, row, col + 1)


    def MAQ(self, MAQInput, startCorner, endCorner): # That is query function
        start_row, start_col = startCorner
        end_row, end_col = endCorner

        # return min element in given region as range from precompted table
        return self.precomputedTable[(len(MAQInput[0]) * start_row) + start_col][(len(MAQInput[0]) * end_row) + end_col]

    def main(self, MAQInput, queries):

        # Think the 2D array as 1D array, However, don't create a 1D array. Use index only for thinking
        startCorner = (0, 0)
        endCorner = (len(MAQInput) - 1, len(MAQInput[0]) - 1)
        corners = (startCorner, endCorner)
        self.full_preprocessing(MAQInput, corners)

        print("FullPreprocessing:\n")
        for left, right in queries:
            minValue = self.MAQ(MAQInput, left, right)
            print(f"MAQ({left}, {right}) =", minValue)
        print("\n---------------------------------")


# ----------------------------------------------------------------------------------------------------------------------


class SparseTable:

    def __init__(self):
        self.sparseTable = None # Store the sparse table to use in future.

    def createSparseTable(self, MAQInput):
        # m = row number of MAQInput
        # n = column number of MAQInput
        # numberOfElements = m * n
        # row = numberOfElements
        # --------------------------
        # column = int(log 2 base numOfElements) + 1

        # Initializing sparse table as 2D array (row x column)
        sparseTable = np.full((len(MAQInput) * len(MAQInput[0]), int(np.log2(len(MAQInput) * len(MAQInput[0]))) + 1), None)
        sparse_table_row_index = 0

        # fill each row of 0th column of sparse table with MAQInput elements by using for loop
        for row in MAQInput:
            for data in row:
                sparseTable[sparse_table_row_index][0] = data
                sparse_table_row_index += 1

        self.sparseTable = sparseTable
        # sparse table is ready to precompute

    def precompute(self, row, col, indexOfLastRow): # it precomputes other cells of sparse table recursively
        temp_row = row # temp row is fixed as 0
        second_row = temp_row + (2 ** col) # second row is two to the power of current column

        # While second row is less than or equal to last row, continue precomputing process.
        while second_row <= indexOfLastRow:
            # compare two elements and select min, then fill sparseTable[temp_row][col + 1] with min value
            self.sparseTable[temp_row][col + 1] = min(self.sparseTable[temp_row][col], self.sparseTable[second_row][col])
            temp_row += 1
            second_row = temp_row + (2 ** col)

        # if current column + 1 == last column of the sparse table, that means precomputing process finished
        if col + 1 == len(self.sparseTable[0]) - 1:
            return

        # Otherwise, continue precomputing process by increasing current column by 1 and by decreasing temp_row by 1
        return self.precompute(row, col + 1, temp_row - 1)

    def MAQ(self, MAQInput, startCorner, endCorner): # That is query function.
        start_row, start_col = startCorner
        end_row, end_col = endCorner
        numOfCol = len(MAQInput[0])

        # Think 2D array as 1D array, but don't create
        start = (start_row * numOfCol) + start_col # start index of 1D virtual array
        end = (end_row * numOfCol) + end_col # end index of 1D virtual array

        t = int(np.log2(end - start + 1)) # int(log 2 base number of elements of 1D virtual array)

        return min(self.sparseTable[start][t], self.sparseTable[end - (2**t) + 1][t]) # return min value of th given region as a range

    def main(self, MAQInput, queries):

        self.createSparseTable(MAQInput)
        self.precompute(0, 0, len(self.sparseTable) - 1)

        print("Sparse Table:\n")
        for left, right in queries:
            minValue = self.MAQ(MAQInput, left, right)
            print(f"MAQ(({left}), {right}) =", minValue)
        print("\n---------------------------------")


# ----------------------------------------------------------------------------------------------------------------------


class Node:

    def __init__(self, data, index):
        self.data = data # data of node
        self.leftNode = None # left child of current node as Node object
        self.rightNode = None # # right child of current node as Node object
        self.index = index # index of data in 1D virtual array

class CartesianTree:

    def __init__(self):
        self.root = None # Store the root of Cartesian Tree to use in future.

    def constructCartesianTree(self, MAQInput): # That creates Cartesian Tree data structure
        # Initializing an empty stack and None root
        stack = []
        root = None

        index = 0
        # For loop constructs whole tree
        for row in MAQInput:
            for element in row:
                # For each element in 2D array, create a Node object
                node = Node(element, index)
                # While stack is not empty and last data of stack is bigger than current element, pop the stack and assign the poped Node to left child of current node
                while len(stack) > 0 and stack[-1].data > element:
                    node.leftNode = stack.pop()
                # if stack is not still empty, current node is assigned to right child of last element of stack
                if len(stack) > 0:
                    stack[-1].rightNode = node
                # If stack is empty now, That means current node is root.
                else:
                    root = node
                # push the stack created Node object
                stack.append(node)
                index += 1

        return root

    def searchTree(self, start_index, end_index): # That returns the min value of given region as range

        subTree = [] # that consist paths from root to start node and end node
        search_index = start_index
        # For loop finds these paths as a list and append subTree list
        for i in range(2):
            current_node = self.root
            path = [current_node]
            while current_node.index != search_index:
                if search_index > current_node.index:
                    current_node = current_node.rightNode
                else:
                    current_node = current_node.leftNode
                path.append(current_node)
            subTree.append(path)
            search_index = end_index

        path1 = subTree[0]
        path2 = subTree[1]
        temp = {len(path1): path1, len(path2): path2}
        minValue = None

        # For loop finds intersection of these two paths
        for index in range(min(len(path1), len(path2))):
            if path1[index] is not path2[index]:
                minValue = path1[index - 1]
                break
        # if minValue is None, that means intersection of two path is root of cartesian tree
        if minValue is None:
            minValue = temp[min(len(path1), len(path2))][-1]

        return minValue


    def MAQ(self, MAQInput, startCorner, endCorner): # That is query function
        # Initializing variables
        start_row, start_col = startCorner
        end_row, end_col = endCorner
        numOfCol = len(MAQInput[0])
        start = (start_row * numOfCol) + start_col
        end = (end_row * numOfCol) + end_col
        # call searchTree and find min value of given region as range (start, end)
        return self.searchTree(start, end).data


    def main(self, MAQInput, queries):

        self.root = self.constructCartesianTree(MAQInput)
        print("Cartesian Tree:\n")
        for left, right in queries:
            minValue = self.MAQ(MAQInput, left, right)
            print(f"MAQ({left}, {right}) =", minValue)
        print("\n---------------------------------")

def readInputFile(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip().split() # extract \n
        num_rows, num_cols = map(int, first_line)
        array2D = np.full((num_rows, num_cols), None)
        lines = file.readlines()
        for i in range(len(lines)):
            row = [int(element) for element in lines[i].strip().split(" ")]
            array2D[i] = row

    return array2D


MAQInput = readInputFile("input.txt") # returns a 2D input array

# create query
query1 = ((0, 0), (5, 6))
query2 = ((0, 0), (0, 6))
query3 = ((2, 2), (3, 3))

queries = [query1, query2, query3] # append the query to queries list

fullPreProcessing = FullPreProcessing()
fullPreProcessing.main(MAQInput, queries)
del fullPreProcessing

sparseTable = SparseTable()
sparseTable.main(MAQInput, queries)
del sparseTable

cartesianTree = CartesianTree()
cartesianTree.main(MAQInput, queries)
del cartesianTree



