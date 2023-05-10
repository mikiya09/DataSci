
# Uninformed Search 
# ------------------
# uniform cost search class 
# can't find the exact path, only found the total cost of the shortest path

# Uniform Cost Search: (according to priority queue)
# 1) a type of greedy algorithm
# 2) make locally optimal choice at each expand (based on those node added to the priority queue)
# 3) meaning: choose the next node who has the lowest cost in the queue to expand
# 4) the node to expand each time is not necessarily in the straight order
# 5) guarantee to find the globally optimal path
# =============================== class ======================================
class Graph:
    def __init__(self):
        self.graph = {}         # create dictionary for storing vertexes and weight
        self.weights = {}

        self.path = []
        self.pq = {}            # priority queue in the uniform cost search
        self.visited = []
        self.optimal_cost = 0
        

    # return weight value with key-mapping in weights dictionary 
    def get_cost(self, from_node, to_node):
        return self.weights[(from_node + to_node)]

    # auto build a map and weight dictionary based on Romania map
    # this is an undirected graph, so it can go back(revisited the passed vertex)
    def auto_path(self):
        # didn't include cities after Bucharest right now
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
        # distance use int value: find a better way please, this is too exhausting
        self.weights = {
            'AradZerind': 75, 'ZerindArad': 75,
            'AradSibiu': 140, 'SibiuArad': 140, 
            'AradTimisoara': 118,'TimisoaraArad': 118,
            'ZerindOradea': 71, 'OradeaZerind': 71,
            'OradeaSibiu': 151,'SibiuOradea': 151,
            'SibiuFagaras': 99, 'FagarasSibiu': 99,
            'FagarasBucharest': 211, 'BucharestFagaras': 211,
            'SibiuRimnicu Vilcea': 80,'Rimnicu VilceaSibiu': 80,
            'Rimnicu VilceaPitesti': 97, 'PitestiRimnicu Vilcea': 97,
            'PitestiBucharest': 101, 'BucharestPitesti': 101,
            'Rimnicu VilceaCraiova': 146, 'CraiovaRimnicu Vilcea': 146,
            'CraiovaPitesti': 138, 'PitestiCraiova': 138,
            'TimisoaraLugoj': 111, 'LugojTimisoara': 111,
            'LugojMehadia': 70,'MehadiaLugoj': 70,
            'MehadiaDrobeta': 75, 'DrobetaMehadia': 75,
            'DrobetaCraiova': 120, 'CraiovaDrobeta': 120,

            'AradArad': 0,
            'ZerindZerind': 0,
            'SibiuSibiu': 0,
            'TimisoaraTimisoara': 0,
            'OradeaOradea': 0,
            'LugojLugoj': 0,
            'MehadiaMehadia': 0,
            'DrobetaDrobeta': 0,
            'CraiovaCraiova': 0,
            'RimnicuVilcea Rimnicu Vilcea': 0,
            'PitestiPitesti': 0,
            'BucharestBucharest': 0,
            'FagarasFagaras': 0
        }

    # clean up edges and weights 
    def empty(self):
        self.graph = {} 
        self.weights = {}
        self.pq = {}

    # display prior queue 
    def display_pq(self):
        print(self.pq)

    # update priority queue based on the new vertex and cost given
    def update_pq(self, vertex, cost):

        if vertex in self.visited:
            if vertex not in self.pq:           # if the node has been visited and is in the PQ, do nothing
                pass 
            else:
                if cost < self.pq[vertex]:      # if visited and in the PQ
                    self.pq[vertex] = cost      # update the cost if it's smaller
                elif cost > self.pq[vertex]:    # else do nothing
                    pass 
        else:
            self.pq[vertex] = cost              # if not visited yet, update as what it is


    # pop the first vertex in priority queue 
    # if two keys has the same minimum value, the first in order will be return 
    # here ths .get method is used, 
    # other options like finding the minimum value then check the key with that value found
    def pop(self):
        pVertex = min(self.pq, key=self.pq.get)
        return pVertex


    # remove node from the PQ
    def remove(self, vertex):
        del self.pq[vertex]


    # calculate a path's total cost 
    def calc_path(self, path):
        total_cost = 0
        for i in range(len(path)):
            if i+1 == len(path):
                return total_cost
            else:
                total_cost += self.get_cost(path[i], path[i+1])

    # get all paths, given the following
    # 1) graph 
    # 2) start node 
    # 3) end node 
    def get_all_paths(self, graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            self.path.append(path)
        else:
            for node in graph.graph[start]:
                if node not in path:
                    # you don't need to add self for the recursive call in self.function
                    self.get_all_paths(graph, node, end, path)

        return self.path


    # match the path with total cost
    def match_path(self, graph, start, end, cost):
        # clear what's in the path originally
        self.path = []
        # collect all the paths based on the given start and end 
        self.get_all_paths(graph, start, end)
        for i in self.path:
            if self.calc_path(i) == cost:
                print(f"Path: {i}, Cost: {self.calc_path(i)}")





# ================= main ================
# create Graph() object
g = Graph() 
# adding the path to the graph
g.auto_path()
start = 'Arad'
end = 'Bucharest'


# uniform cost search
def ucs(graph, start, end):
    
    # adding the start node to the PQueue
    if start not in graph.pq: 
        graph.pq[start] = 0
    else:
        pass
    # if the start is the end, then terminate the function
    if start == end:
        print(f"found: {start}, cost: {graph.pq[start]}")
        # update path cost
        graph.optimal_cost = graph.pq[start]
        return 
        # return

    # expand the start node
    for v in graph.graph[start]: 
        # ignore visited nodes
        if v in graph.visited:
            pass
        else:
            # update weight from start to children nodes, and remember to add the weight of start node as well
            weight = graph.get_cost(start, v) + graph.pq[start]
            graph.update_pq(v, weight)

    # adding the visited node to the path
    graph.visited.append(start)
    # remove the node that had been expanded
    graph.remove(start)
    # set the newly smallest node as the start node and start another function call
    ucs(graph, graph.pop(), end)



# run uniform cost search, find the optimal cost
ucs(g, start, end)
# match the path with optimal cost
g.match_path(g, start, end, g.optimal_cost)

