

map = {
    'Arad': ['Zerind', 'Sibiu', 'Timisoara'],
    'Zerind': ['Oradea'],
    'Oradea': ['Sibiu'],
    'Sibiu': ['Fagaras', 'Rimnicu Vilcea'],
    'Fagaras': ['Bucharest'],
    'Rimnicu Vilcea': ['Pitesti', 'Craiova'],
    'Craiova': ['Pitesti'],
    'Pitesti': ['Bucharest'],
    'Bucharest': [],

    'Timisoara': ['Lugoj'],
    'Lugoj': ['Mehadia'],
    'Mehadia': ['Dobreta'],
    'Dobreta': ['Craiova']
}



# Always Think of Tree Structure for DFS and BFS
#               a 
#            /     \ 
#           c       s 
#         /   \   /   \ 
#        t    e   c    i 
#       / \  / \ / \  / \ 
#      y  x  z x d v  f  m
#     . . . . . . . . . . . 



# Depth First Search 
def dfs(graph, vertex, end, visited=set()): 

    # path's direction is determined 
    # 1) depth first search, algorithm will always explore one path to the end 
    # 2) starting from left to right in tree structure
    if vertex == end:
        # print(visited)
        return True 
    visited.add(vertex)
    for neighbour in graph[vertex]:
        # 3) for each node: dfs will explore one path to the end and go back for another path (left -> right)
        if neighbour not in visited and dfs(graph, neighbour, end, visited):
            # 4) here recursive call explore left node path to the end, check from the last call to the first call
            # 5) if found, return true; otherwise do nothing and start checking the right node path
            # 6) unlike bfs could have different search paths, the search pattern for dfs is fix
            # 7) only unique nodes that had been visited stored is enough, so we use set
            # 8) the path will be listed out in this if statement
            print(neighbour)
            return True 
    return False






# Breadth First Search 
def bfs(graph, start, end):
    queue = [start]         # ['Arad']
    visited = [start]       # ['Arad']
    
    # while queue still have value
    while queue:
        vertex = queue.pop(0)
        if vertex == end:
            # display the path for search, 
            # 1) since bfs, the algorithm will only check nodes within this level then go to the next level 
            # 2) the order nodes enqueued is in the order of how graph(dict) is ordered 
            # 3) for key 'Arad': values are in order of 'Zerind', 'Timisoara', 'Sibiu'
            for i in visited:
                print(i)
                if i == end:
                    break
            return True 
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                # 4) when they(those not visited) are added to the queue as neighbour of 'Arad', checking start again 
                # 5) start dequeuing Zerind while adding Zerind's value, then Timisoara's value 
                # 6) when it comes to dequeuing 'Sibiu', loop stop and found 
                # 7) that's the reason when printing out all elements in the queue, there are Oradea and Lugoj
                # 8) so I add an if statement to print only elements up to Sibiu
                # 9) there is a possibility in graph that you might visit a same node twice, here is Sibiu 
                # 10) bfs path is not fixed, so even the same node is visited but it belongs to a different path 
                # 11) so use repeated array(list) for clarity
                visited.append(neighbour)
                queue.append(neighbour)
    return False

    


# bfs(map, 'Arad', 'Bucharest')
bfs(map, 'Arad', 'Bucharest')




