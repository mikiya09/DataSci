

# uniform cost search 
class Graph:
    def __init__(self):
        self.edges = {}         # create dictionary for storing path and weight
        self.weights = {}

    # return city value with key-mapping in edges dictionary 
    def neighbors(self, node):
        return self.edges[node]

    # return weight value with key-mapping in weights dictionary 
    def get_cost(self, from_node, to_node):
        return self.weights[(from_node + to_node)]

    # auto build a map and weight dictionary based on Romania map
    def auto_path(self):
        # didn't include cities after Bucharest right now
        self.edges = {
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
        # distance use int value
        self.weights = {
            'AradZerind': 75,
            'AradSibiu': 140,
            'AradTimisoara': 118,
            'ZerindOradea': 71,
            'OradeaSibiu': 151,
            'SibiuFagaras': 99,
            'FagarasBucharest': 211,
            'SibiuRimnicu Vilcea': 80,
            'Rimnicu VilceaPitesti': 97,
            'PitestiBucharest': 101,
            'Rimnicu VilceaCraiova': 146,
            'CraiovaPitesti': 138,

            'TimisoaraLugoj': 111,
            'LugojMehadia': 70,
            'MehadiaDobreta': 75,
            'DobretaCraiova': 120
        }

    # prompt use to enter nodes and weights 
    def addVertex(self):
        edge = input("Enter a node: ")
        self.edges[edge] = []
        weight = input("Enter a weight: ")
        self.weights[edge] = [weight]

    # clean up edges and weights 
    def empty(self):
        self.edges = {} 
        self.weights = {}


#
g = Graph() 
# g.prompt()
# g.prompt()
g.auto_path()

# greedy algorithm: uniform cost search 
# make locally optimal choice at each step 
# meaning: choose the next vertex to visit based on the minimum cost among nodes at the current level 
# not necessarily get the shortest path globally, but shortest path for every two nodes 

def ucs(graph, start, end):
    queue = [start]
    visited = [start]
    total = 0

    while queue:
        vertex = queue.pop(0)
        if vertex == end:
            # print(visited)
            # print(queue)
            print(f'total distance: {total}')
            return total
        
        result = float('inf')
        holder = ''
        # for num, node in enumerate(graph.edges[vertex]):
        for node in graph.edges[vertex]:
            weight = graph.get_cost(vertex, node)
            if weight < result:
                result = weight
                holder = node
                print(holder)
        if holder not in visited:
            visited.append(holder) 
            queue.append(holder)
            print(f"{vertex} -> {holder}: {result}")
            total += result

print(f"shortest distance based on Uniform Cost Search: {ucs(g, 'Arad', 'Bucharest')}")


