

# Informed(heuristic) Search --> we have additional knowledge about the search task 
# Greedy best-first search 
# 1) h(n): heuristic is the function for additional knowledge 
#    - h_SLD = straight line distance heuristic
# 2) g(n): is the cost to reach the node
#    - path cost such as 140 from Arad to Sibiu 
# 3) while uniform cost search use g(n), greedy use h(n) function as the cost estimate 
# 4) A* search use the combination of g(n) and h(n)



class Graph:
    def __init__(self):
        self.graph = {}         # create dictionary for storing vertexes and weight
        self.weights = {}
        self.path = []
        

    def get_cost(self, vertex):
        return self.weights[vertex]
    
    def auto_path(self):
        # graph
        self.graph = {
            'Arad': ['Zerind', 'Sibiu', 'Timisoara'],
            'Zerind': ['Arad', 'Oradea'],
            'Oradea': ['Zerind', 'Sibiu'],
            'Sibiu': ['Arad', 'Oradea', 'Fagaras', 'Rimnicu Vilcea'],
            'Fagaras': ['Sibiu', 'Bucharest'],
            'Rimnicu Vilcea': ['Sibiu', 'Pitesti', 'Craiova'],
            'Craiova': ['Drobeta', 'Pitesti'],
            'Pitesti': ['Rimnicu Vilcea', 'Bucharest'],
            'Bucharest': [],

            'Timisoara': ['Arad', 'Lugoj'],
            'Lugoj': ['Timisoara', 'Mehadia'],
            'Mehadia': ['Lugoj', 'Drobeta'],
            'Drobeta': ['Mehadia', 'Craiova']
        }

        # h(n) function 
        self.weights = {
            'Arad': 366,
            'Bucharest': 0,
            'Craiova': 160,
            'Drobeta': 242,
            'Eforie': 161,
            'Fagaras': 176,
            'Giurgiu': 77,
            'Hirsova': 151,
            'Iasi': 226,
            'Lugoj': 244,
            'Mehadia': 241,
            'Meamt': 234,
            'Oradea': 380,
            'Pitesti': 100,
            'Rimnicu Vilcea': 193,
            'Sibiu': 253,
            'Timisoara': 329,
            'Urziceni': 80,
            'Vaslui': 199,
            'Zerind': 374
        }

def greedy(graph, start, end):

    # if the list is empty, add the first(start) vertex to it
    if not graph.path:
        graph.path.append(start)

    # if the start vertex has the cost of 0, meaning the start is also the destination
    if graph.weights[start] == 0:
        print(graph.path)
        return 

    # otherwise expand the start node, and collect the smallest cost child based on h(n) function
    children_node = {}
    for v in graph.graph[start]:
        children_node[v] = graph.get_cost(v)

    # find the smallest vertex and add it to the path
    smallest = min(children_node, key=children_node.get)
    graph.path.append(smallest)
    # recursivecall
    greedy(graph, smallest, end)

# main
g = Graph()
g.auto_path()

greedy(g, 'Arad', 'Bucharest')
