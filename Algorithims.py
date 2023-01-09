from queue import Queue             # Implementing a Queue
from collections import defaultdict # Implements a graph

'''
    Bubble sort works by repeatedly swapping the adjacent elements. Not suitable for large datasets
    since time complexity is bad. Simplist of sorting algorithims.

    Time: O(N^2)
    Space: O(1)
'''
def bubbleSort(arr):
    n = len(arr)                    # Length of the array
    # Traverse through all array elements
    for i in range(n):
 
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

'''
    Merge sort is a divide and conquer algorithm. It splits the array into halves recursively until 1 element remains
    Each half is then sorted and merged back together. 

    Time: O(n*log(n))
    Space: O(n)
'''
def mergeSort(arr):

    if len(arr) > 1:
        # Splits the array into sub arrays
        left_arr = arr[:len(arr)//2]
        right_arr = arr[len(arr)//2:]

        # Recursively calls mergeSort until 1 element is left in the array
        mergeSort(left_arr)
        mergeSort(right_arr)

        i = 0           # Left array index
        j = 0           # Right array index
        k = 0           # Original array index

        # Iterate through the left most  element in each array. Whichever is less, we put into
        # the original array. What ever is left over we append to the end of the array after 
        # the loop
        while (i < len(left_arr)) and (j < len(right_arr)):
            if left_arr[i] < right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                j += 1
            
            k +=1

        # Whatever is left over from the original array will be appended to the ouput array
        while i < len(left_arr):
            arr[k] = left_arr[i]
            i += 1
            k += 1

        while j < len(right_arr):
            arr[k] = right_arr[j]
            j += 1
            k += 1


'''
    Quick sort is a divide and conquer algorithm. It picks an element as a pivot and then 
    partitions the given array around the picked pivot.

    Time: O(n*log(n)) -> worst case is O(n^2)
    Space: O(log(n))
'''
def quickSort(arr, left, right):
    # Subarray contains at least 2 elements
    if left < right:
        partition_pos = partition(arr, left, right)
        quickSort(arr, left, partition_pos-1)
        quickSort(arr, partition_pos+1, right)

# Returns index of the pivot element after the first step of quicksort
def partition(arr, left, right):
    i = left
    j = right - 1
    pivot = arr[right]

    while i < j:
        # Checks for i to not be at the end of the array and less then the pivot
        while i < right and arr[i] < pivot:
            i += 1
        # Checks for j to not be at the beginning of the array and greater then the pivot
        while j > left and arr[j] >= pivot:
            j -= 1

        # Checks to see if the indexs swapped. If not the elements at each index gets swapped
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    
    # If the element at i is greater then the pivot, the elements must be swapped
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]

    # Index to split quicksort recursively
    return i      

'''
    Binary search works on sorted arrays by dividing the array in half until the desired value is found

    Time: O(log(n))
    Space: O(log(n))
'''

def binary_search(arr, l, r, x):

    # Checks to see if there are more values in the array to check
    if r >= l:
        # Finds the initial mid point of the array
        mid = (l + (r-1))//2

        # Check to see if the mid point is desired value
        if arr[mid] == x:
            return mid

        # Checks to see if the desired value is greater or less then mid value
        if x > arr[mid]:
            return binary_search(arr, mid + 1, r, x)
        else:
            return binary_search(arr, l, mid - 1, x)
    
    else:
        return -1




'''
    FIFO
'''
def queue_implementation():
    # Initializing a queue
    # maxsize gives number of items allowed in the queue
    q = Queue(maxsize = 3)
    
    # qsize() returnss the number of items in the queue 
    print(q.qsize()) 
    
    # Adding of element to queue
    q.put('a')
    q.put('b')
    q.put('c')
    
    # Return Boolean for Full 
    print("\nFull: ", q.full()) 
    
    # Removing element from queue
    print("\nElements dequeued from the queue")
    print(q.get())
    print(q.get())
    print(q.get())
    
    # Return Boolean for Empty 
    # Queue 
    print("\nEmpty: ", q.empty())
    
    q.put(1)
    print("\nEmpty: ", q.empty()) 
    print("Full: ", q.full())
    print(q.get())

    
    # This would result into Infinite 
    # Loop as the Queue is empty. 
    # print(q.get())

    # To not wait when enque on a full queue and deque on an empty queue:
    # put_nowait(item) – Put an item into the queue without blocking. If no free slot is immediately available, raise QueueFull.
    # get_nowait() – Return an item if one is immediately available, else raise QueueEmpty.

    return None

'''
    3 rules for a binary tree:
    1) At most 2 children per node
    2) Exactly 1 root
    3) Exactly 1 path between root and any node
'''
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def tree_BFS(root):

    if root == None:
        return []

    q = Queue()                                 # Queue to hold nodes
    arr = []
    q.put(root)                                 # Puts intial value into queue

    # Iterate through tree
    while not q.empty():
        current_node = q.get()                  # Gets the first value out of the queue
        arr.append(current_node.val)

        if current_node.left:
            q.put(current_node.left)
        if current_node.right:
            q.put(current_node.right)

    return arr

def tree_DFS(root):

    # Checks for empty tree
    if root == None:
        return []

    stack = [root]                              # Creates stack to hold nodes of the tree
    arr = []

    # Loops through the whole tree until every node has been visited
    while len(stack) != 0:
        current_node = stack.pop()              # Pops elements off the stack

        print(current_node.val)
        arr.append(current_node.val)

        # Checks if child nodes exist. Adds the nodes to the stack if present
        if(current_node.right):
            stack.append(current_node.right)
        if(current_node.left):
            stack.append(current_node.left)

    return arr

def recursive_tree_DFS(root):
    if root == None:
        return []

    leftVals  = recursive_tree_DFS(root.left)
    rightVals = recursive_tree_DFS(root.right)

    arr = [root.val]
    arr.extend(leftVals)
    arr.extend(rightVals)

    return arr



class Graph:
 
    # Constructor
    def __init__(self):
 
        # default dictionary to store graph
        self.graph = defaultdict(list)
 
    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # Returns all node/vertex connections
    def generate_edges(self):
        edges = []
    
        # for each node in graph
        for node in self.graph:
            
            # for each neighbour node of a single node
            for neighbour in self.graph[node]:
                
                # if edge exists then append
                edges.append((node, neighbour))
        return edges

    '''

    Steps for BFS:

    1) Declare a queue and insert the starting vertex.
    2) Initialize a visited array and mark the starting vertex as visited.
    3) Follow the below process till the queue becomes empty:
        3.1) Remove the first vertex of the queue.
        3.2) Find all the neighbors of the vertex
        3.3) If the neighbor vertex is not visited
            3.3.1) Enque the vertex and mark the vertex as visited

    Time Complexity: O(V+E)
    Space: O(V)

    '''
    # Function to print a BFS of graph
    def BFS(self, s):
        q = Queue()                                 # Holds all neighboring nodes      
        visited = [s]                               # Holds all the visited nodes. The starting node is initialized as visited
    
        q.put(s)                                    # Enqueue the first node

        while not q.empty():
            node = q.get()                          # Removes the node from the queue
            print(node, end=" ")

            for i in self.graph[node]:
                if i not in visited:
                    q.put(i)                        # Enque the node so that we can visit it's neighbors
                    visited.append(i)               # Mark the node as visited so we don't visit it again


    '''

    Steps for a DFS:
    1) Create a functions that takes the indexs of the node and a visited array
    2) In the function, first thing to do is see if the node has been visited
    3) If the node has not been visited, add it to the visited array
        3.1) Iterate through the nodes neighbors
        3.2) Recursively call the function

    Time Complexity: O(V+E)
    Space: O(V)

    '''
    # Function to print a DFS of graph
    def DFS(self, s, visited):

        # Checks to see if the node has been visited
        if s not in visited:
            visited.append(s)               # Mark the node as visited

            print(s, end=' ')

            # Visit the neighbors of the node
            for i in self.graph[s]:
                self.DFS(i, visited)       # Recursively calls DFS to visit the next node until each node has been visited in the path




if __name__ == '__main__':
    # Queue implementation
    # queue_implementation()

    # Create a graph given in
    # the above diagram
    g = Graph()
    # g.addEdge(0, 1)
    # g.addEdge(0, 2)
    # g.addEdge(1, 0)
    # g.addEdge(1, 2)
    # g.addEdge(1, 3)
    # g.addEdge(2, 0)
    # g.addEdge(2, 1)
    # g.addEdge(2, 4)
    # g.addEdge(3, 1)
    # g.addEdge(3, 4)
    g.addEdge(5, 3)
    g.addEdge(5, 7)
    g.addEdge(3, 2)
    g.addEdge(3, 4)
    g.addEdge(7, 8)
    g.addEdge(4, 8)


    print(g.generate_edges())

    # Runs BFS on graph
    # g.BFS(5)

    # Runs DFS on graph
    # visited = []
    # g.DFS(5, visited)

    a = Node('a')
    b = Node('b')
    c = Node('c')
    d = Node('d')
    e = Node('e')
    f = Node('f')

    a.left = b
    a.right = c
    b.left = d
    b.right = e
    c.right = f

    # print(tree_DFS(a))
    # # print(tree_DFS(None))

    # print(recursive_tree_DFS(a))

    # print(tree_BFS(a))


    # arr = [5, 1, 4, 2, 8]
    arr = [2,3,5,1,7,4,4,4,2,6,0]

    # bubbleSort(arr)

    print(arr)

    # mergeSort(arr)
    quickSort(arr, 0, len(arr) - 1)

    print(arr)

    print(binary_search(arr, 0, len(arr) - 1, 5))
