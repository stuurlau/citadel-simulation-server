######################################################################################################
# transform_functions.py
#
# Liza Roelofsen
#
# transform_agents_to_citadel_graph:
# Transforms a NetworkX graph into a JSON graph.
# Input: the criminal network as in the Replacement Model in NetworkX format.
# Output: same network in JSON file format (to be used by Citadel).
# 
# transform_citadel_to_agents_graph:
# Transforms a JSON graph into a NetworkX graph.
# Input: the criminal network as in the Replacement Model in JSON file format (as used by Citadel).
# Output: same network in NetworkX format (to be used by the Replacement Model).
######################################################################################################

# ASSUMPTIONS:
# - Node and edge attributes are floats between 0 and 1
# - Node and edge tags are strings
# - All strings are lowercase with underscores instead of spaces
# - The graph is undirected

# Imports
import networkx as nx
import numpy as np
import copy
import string

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from A3 import fred

Node = fred.Node
_role_kingpin_main_str = fred._role_kingpin_main_str
_node_tags_values = fred._node_tags_values
_node_attribute_values = fred._node_attribute_values

# Functions
def check_graph_statistics(graph) -> dict:
    return {"Number of Nodes": graph.number_of_nodes(),
            "Number of Edges": graph.number_of_edges(),
            "Density": round(nx.density(graph), 2),
            "Average Degree": round(sum(dict(graph.degree()).values())/graph.number_of_nodes(), 2),
            # "Average Clustering Coefficient": round(nx.average_clustering(model.graph), 2),
            # "Average Shortest Path Length": round(nx.average_shortest_path_length(model.graph), 2), # Does not work if not all nodes are connected
            # "Diameter": nx.diameter(model.graph),
            # "Efficiency": round(nx.global_efficiency(graph), 2), # Does not work for directed graphs
            "Number of Components": nx.number_strongly_connected_components(graph),
            "Most Central Nodes (Top 5)": (' --> ').join([str(id) for id, node in sorted(nx.in_degree_centrality(graph).items(), key= lambda x: x[1], reverse=True)[:5]])
            }

def get_attrs(agent, hidden_attributes = []) -> dict:
    '''Returns the attributes of an agent in a dictionary.'''

    visible_attributes = {key: value for key, value in agent.__dict__.items() if key not in hidden_attributes}

    attrs = {string.capwords(str(key).replace('_', ' ')): string.capwords(str(value).replace('_', ' ')) for key, value in visible_attributes.items() if isinstance(value, (int, float, str, bool, list))}

    for key, value in visible_attributes.items():
        if isinstance(value, dict):
            value = {string.capwords(str(key).replace('_', ' ')): (string.capwords(str(value).replace('_', ' ')) if isinstance(value, str) else value) for key, value in value.items()}
            attrs.update(value)

        if isinstance(value, Node):
            value = {string.capwords(str(key).replace('_', ' ')): (string.capwords(str(value).replace('_', ' ')) if isinstance(value, str) else value) for key, value in value.__dict__.items() if isinstance(value, (int, float, str, bool, list))}
            attrs.update({string.capwords(str(key).replace('_', ' ')): value})

        # if isinstance(value, set):
        #     if [item for item in value if isinstance(item, Node)] != []:
        #         value = [item.unique_id for item in value if isinstance(item, Node)]
        #     else:
        #         value = list(value)
        #     attrs.update({string.capwords(str(key).replace('_', ' ')): value})

    return attrs

def transform_agents_to_Citadel_graph(model, include=[], pos=None, fixed_node_pos=None, conclave=None, debug=False):
    '''
    This functions takes the NetworkX graph and transforms it to the JSON format.
    Additionally, it deletes and adds the attributes that we want for the simulation on Citadel
    compared to the simulation within python itself.
    '''

    graph = model.graph # shorthand
    agents = model.agents # shorthand
    
    include_options = ['SNA metrics', 'position', 'meeting', 'fitness', 'distance']
    if 'all' in include:
        include = include_options

    # Check if the provided include values are valid options
    for item in include:
        if item not in include_options:
            raise ValueError(f'{item} is not a valid option for include, options are: centralities, clustering, position, meeting, all')
    
    if debug:
        print(f'Including: {include}')

    # Initialize the nodes and edges
    nodes = []; edges = []

    if 'SNA metrics' in include:
        if nx.is_strongly_connected(graph):
            center_nodes = nx.center(model.graph)
        else:
            center_nodes = []
        
        # communities = nx.community.greedy_modularity_communities(graph)

        # Get centrality measures
        indegree = nx.in_degree_centrality(graph)
        outdegree = nx.out_degree_centrality(graph)
        close = nx.closeness_centrality(graph)
        between = nx.betweenness_centrality(graph)

    # TODO: this is not used now
    # if 'position' in include:
    #     if pos is None:
    #         fixed = None
    #         threshold = 1e-4 #0.01
    #         k = None # k=1000 is too limiting for initial conditions, but afterward, quite nice.
    #         pos = nx.spring_layout(graph, pos = pos, weight = 'Trust', scale = (225 * np.log(len(agents))), k=k, threshold = threshold, fixed = fixed)

        # else:
        #     fixed = fixed_node_pos # NOTE: in the future might want more nodes fixed --> [int(id) for id in pos.keys()]
        #     threshold = 0.9
        #     k = 1000

        # TODO: include choice of layout by user? include: keep position fixed (of one node to be given as an integer).
        layout_options = ['_sparse_fruchterman_reingold', '_fruchterman_reingold', 'kamada_kawai_layout', 'shell_layout', 'spring_layout', 'spectral_layout']
        # pos = nx.layout('spring_layout', graph, pos=pos, weight='Trust', scale = (225 * np.log(len(agents))), fixed=fixed, threshold=threshold, k=k) # TODO: check if this works

    if 'distance' in include:
        # Get the distance to the removed kingpin
        dists = nx.shortest_path_length(model.graph, source=[node.unique_id for node in model.agents.values() if node.role_str() == _role_kingpin_main_str][0])

    if debug:
        print('after initialisation is handled')

    # Add the nodes to the graph
    for node in agents.values():

        id = int(node.__dict__['unique_id']) # shorthand for the id of the node
        hidden_attributes = [key for key in node.__dict__ if key not in ['tags', 'attributes', 'fitness', 'description', 'unique_id']] +  [string.capwords(str(key).replace('_', ' ')) for key in node.__dict__ if key not in ['tags', 'attributes', 'fitness', 'description', 'unique_id']]

        if debug:
            print('not the id')

        attrs = get_attrs(node)

        if debug:
            print('got the attributes')

        if 'SNA metrics' in include:
            # Get the clustering coefficient
            # attrs['Clustering'] = nx.clustering(graph, id)
            
            # Get the degree of the node
            attrs['_In Degree'] = graph.in_degree[id]
            attrs['_Out Degree'] = graph.out_degree[id]
            
            # Get the centrality measures
            attrs['_Centrality (In Degree)'] = indegree[id]
            attrs['_Centrality (Out Degree)'] = outdegree[id]
            attrs['_Centrality (Closeness)'] = close[id]
            attrs['_Centrality (Betweenness)'] = between[id]

            # Get the center nodes
            # attrs['_Center Node'] = True if id in center_nodes else False

            # Get the community of the node
            # for i, community in enumerate(communities):
            #     if id in community:
            #         attrs['_Community'] = i

            if debug:
                print('got the SNA metrics')

        if 'meeting' in include:
            if conclave != None and id in conclave:
            # Get the conclave agent and give them the 'In Conclave' attribute
            # if node[0] in conclave:
                attrs['_In Conclave'] = True
            else:
                attrs['_In Conclave'] = False

            if debug:
                print('got the conclave')

        if 'fitness' in include and model.vn != None:
            # Add the fitness of the agents to the nodes
            attrs['Fitness'] = round(node.fitness(model.vn),2)

            if debug:
                print('got the fitness')

        # for key, value in node.__dict__.items():
        #     if key not in attrs:
        #         print(key)
        #         key = string.capwords(str(key).replace('_', ' '))
        #         value = string.capwords(str(value).replace('_', ' ')) if isinstance(value, str) else value
        #         attrs[key] = value

        if debug:
            print('before the extra attributes')
        
        # # TODO: CHECK IF THIS IS STILL NECESSARY
        # if 'distance_to_removed_kingpin' in node.__dict__ and 'Distance To Removed Kingpin' not in attrs:
        #     # print("distance", node.__dict__['distance_to_removed_kingpin'])
        #     attrs['_Distance To Removed Kingpin'] = node.__dict__['distance_to_removed_kingpin']
        #     # print('necessary to still add the distance')
        
        # if 'Distance To Removed Kingpin' in attrs and attrs['Distance To Removed Kingpin'] is not None and attrs['Distance To Removed Kingpin'] == 'Inf': # NOTE: was np.infty
        #     attrs['_Distance To Removed Kingpin'] = -1

        if 'distance' in include:
            # Get the distance to the removed kingpink
            attrs['_Distance To Removed Kingpin'] = dists[id] if id in dists and dists[id] != np.infty else -1
        else:
            attrs['Distance To Removed Kingpin'] = node.__dict__['distance_to_removed_kingpin'] if node.__dict__['distance_to_removed_kingpin'] != np.infty or node.__dict__['distance_to_removed_kingpin'] == 'Inf' else -1

        if debug:
            print("distance added:", attrs)

        attrs['Searching Replacement For'] = str(get_attrs(node.__dict__['searching_replacement_for'], hidden_attributes=hidden_attributes)) if node.__dict__['searching_replacement_for'] is not None else None
        # if 'Searching Replacement For' not in attrs:
        #     print('searching replacement for')
        #     print(node.__dict__['searching_replacement_for'])
        #     if 'searching_replacement_for' in node.__dict__ and node.__dict__['searching_replacement_for'] is not None:
        #         print('node found, adding to attrs')
        #         attrs['Searching Replacement For'] = str(get_attrs(node.__dict__['searching_replacement_for']))
        #     else:
        #         print('node not found, adding None')
        #         attrs['Searching Replacement For'] = None

        if debug:
            print("replacement added:", attrs)


        # attrs['_Aware Of Candidates'] = [agent for agent in list(node.__dict__['aware_of_candidates'])]

        # if 'Aware Of Candidates' not in attrs and 'aware_of_candidates' in node.__dict__:
        attrs['_Aware Of Candidates'] = []
        for agent in node.__dict__['aware_of_candidates']:
            if isinstance(agent, int):
                attrs['_Aware Of Candidates'].append(agent)
            else:
                attrs['_Aware Of Candidates'].append(agent.__dict__['unique_id'])

        # if len(attrs['_Aware Of Candidates']) == 0:
        #     del attrs['_Aware Of Candidates']

        # Check is id and unique_id are the same
        if debug:
            print("aware of candidates added:", attrs)
            print('got through the extra attributes')
            
            if str(id) != str(attrs['Unique Id']):
                print(f"id ({id}) and unique_id ({attrs['Unique Id']}) are not the same for some reason.")
            if str(id) != str(attrs['Pos']):
                print(f"id ({id}) and pos ({attrs['Pos']}) are not the same for some reason.")
    
        # Remove the attributes that are not needed
        if 'Unique Id' in attrs:
            del attrs['Unique Id']
        if 'Pos' in attrs:
            del attrs['Pos']
        # if 'position' in include:
        #     del attrs['X']
        #     del attrs['Y']

        # Sort the attributes
        attrs = {('_' + k if k in hidden_attributes else k): attrs[k] for k in attrs}
        attrs = {k: attrs[k] for k in sorted(attrs)}

        if debug:
            print('sorted the attributes')

        # ALTERNATIVE ORDER: according to importance for the model
        # desired_order_dict = ['Business Role', 'Activity', 'Mindset', 'Criminal Capital', 'Financial Capital', 'Violence Capital', 'Centrality (Degree)', 'In Conclave', 'Distance To Removed Kingpin', 'Aware Of Candidates', 'Searching Replacement For', 'Dangling']
        # attrs_sorted = sorted(attrs, key=lambda x: desired_order_dict.index(x) if x in desired_order_dict else len(desired_order_dict))
        # attrs = {k: attrs[k] for k in attrs_sorted}

        # Get the position of the node
        # print('include',include)
        if 'position' in include:
            # add node
            nodes.append({
            "attributes": attrs,
            "id": id,
            "position": pos[id]#{'x': pos[id][0], 'y': pos[id][1]}
            })
        else:
            # add node
            nodes.append({
            "attributes": attrs,
            "id": id,
            })
            
    if debug:
        print('after nodes are handled')
    
    # Add the edges to the graph
    for edge in graph.edges(data=True):
        
        attrs = {string.capwords(str(key).replace('_', ' ')): string.capwords(str(value).replace('_', ' ')) for key, value in edge[2].items()}
        
        if "Id" in attrs:
            del attrs["Id"]

        # add edge
        edges.append({
        "attributes": attrs,
        "source": str(edge[0]),
        "target": str(edge[1]),
        "id": str(edge[0]) + '-' + str(edge[1])
        })

    if debug:
        print('after edges are handled')
    
    return nodes, edges

def transform_Citadel_to_agents_graph(model, nodes, edges, debug=False):
    '''
    This functions takes the nodes and edges from Citadel and transforms it to the graph (and agent instances) as used by the model.
    '''
    
    if debug:
        print('starting to the transform_Citadel_to_agents_graph function')
    
    tags        = [string.capwords(str(tag).replace('_', ' ')) for tag in _node_tags_values.keys()]
    attributes  = [string.capwords(str(attr).replace('_', ' ')) for attr in _node_attribute_values.keys()]
    # variables   = ['Aware Of Candidates', 'Searching Replacement For', 'Distance To Removed Kingpin', 'Dangling']

    # get the graph in the right format (NetworkX)
    network = nx.DiGraph()
    network.add_nodes_from([(int(node['data']['id']), dict(node)) for node in nodes])

    # Create the agents
    for node in network.nodes(data=True):

        # Create the different dictionaries
        tags_dict = {k.replace(' ', '_').lower(): v.replace(' ', '_').lower() for k, v in node[1]['data'].items() if k in tags}
        attributes_dict = {k.replace(' ', '_').lower(): float(v) for k, v in node[1]['data'].items() if k in attributes}
        variables_dict = {k.lstrip('_').replace(' ', '_').lower(): v for k, v in node[1]['data'].items() if k not in tags and k not in attributes and k != 'id'}

        if 'dangling' in variables_dict:
            variables_dict['_dangling'] = variables_dict['dangling']
            del variables_dict['dangling']

        # Get the these variables in Node form
        for key, value in variables_dict.items():
            if value == 'None':
                print("to check line 270 transform function", eval(value), type(eval(value)))
                value = None
            
            elif key == 'searching_replacement_for' and value is not None:
                value = eval(value)

                kingpin_tags_dict = {k.replace(' ', '_').lower(): v.replace(' ', '_').lower() for k, v in value.items() if k in tags}
                kingpin_attributes_dict = {k.replace(' ', '_').lower(): float(v) for k, v in value.items() if k in attributes}

                value = Node(value['Unique Id'], model, tags=kingpin_tags_dict, attributes=kingpin_attributes_dict)
            
            elif key == 'distance_to_removed_kingpin':
                if value == -1:
                    value = np.infty
                elif value != None:
                    value = int(value)

            elif key == 'aware_of_candidates':
                value = set()
            
            elif key  == '_dangling':
                value = eval(value)

            elif isinstance(value, str) and len(value) > 0:
                value = eval(value)
            
            variables_dict[key] = value
        
        if debug:
            print('tags_dict', tags_dict)
            print('attributes_dict', attributes_dict)
            print('variables_dict', variables_dict)

        model.add_agent(node_tags=tags_dict, node_attrs=attributes_dict, node_vars=variables_dict, id=int(node[0]))

    # Add the aware_of_candidates in Node form to the agents
    for node in network.nodes(data=True):
        if '_Aware Of Candidates' in node[1]['data']:
            model.agents[node[0]].aware_of_candidates = set(model.agents[id] for id in node[1]['data']['_Aware Of Candidates'] if id in model.agents)

    # Add the edges to the graph
    for edge in edges:
        edge['attributes'] = {}
        for key, value in edge['data'].items():
            if key not in ['source', 'target']:
                # TODO: make sure it does not crash when extra code is added
                # Currently the attributes are floats
                try:
                    value = float(value)
                # And the tags are strings
                except:
                    value = value.replace(' ', '_').lower()

                edge['attributes'][key.replace(' ', '_').lower()] = value

    model.graph.add_edges_from([[int(edge['data']['source']), int(edge['data']['target']), dict(edge['attributes'])] for edge in edges if int(edge['data']['source']) in model.graph.nodes and int(edge['data']['target']) in model.graph.nodes])

    return model.graph, model.agents