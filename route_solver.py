from cluster_hierarchy_tree import *

def r_solver(tree):
    print(tree.find_node_by_id(2).get_id())

# Get tree
#   Post order traversal
#       Call cluster solver (make distance matrix, solve)
#       
#       Gets approx optimal route
#   