#######################################################################################
# replacement_model_step.py
#
# Liza Roelofsen
#
# Connects the Replacement model to Citadel and performs one simulation step.
# Input: the current graph state in [nodes, edges, params] format.
# Output: the current graph data after being transformed by a simulation step of the 
# Replacement model in [nodes, edges, params] format.
#######################################################################################

# ASSUMPTIONS:
# - Node and edge attributes are floats between 0 and 1
# - Node and edge tags are strings
# - All strings are lowercase with underscores instead of spaces
# - The graph is undirected

# Imports
import copy
import sys
import pathlib
import asyncio
import networkx as nx
import time
import warnings
from pprint import pprint
import string
import json
from transform_functions import transform_agents_to_Citadel_graph, transform_Citadel_to_agents_graph, get_attrs, check_graph_statistics

# sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))
# from citadel.api.runcitadel.src import runcitadel
import traceback

import runcitadel

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from A3 import fred

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from A3.ValueNetworks.vn import import_vn

CocaineNetwork = fred.CocaineNetwork
Node = fred.Node
_potential_roles_for_replacing_role = fred._potential_roles_for_replacing_role
_role_kingpin_candidate_str = fred._role_kingpin_candidate_str
_memory_types = fred._memory_types

_node_tags_values = fred._node_tags_values
_node_attribute_values = fred._node_attribute_values

_business_role_node_tag = fred._business_role_node_tag
_role_kingpin_main_str = fred._role_kingpin_main_str
_role_murderbroker_str = fred._role_murderbroker_str

_mindset_node_tag = fred._mindset_node_tag
_mindset_neutral = fred._mindset_neutral
_mindset_uncertain = fred._mindset_uncertain
_mindset_chaotic = fred._mindset_chaotic

_activity_node_tag = fred._activity_node_tag
_activity_searching = fred._activity_searching

# Functions
def ccrm_sim_step(connection, nodes, edges, params, globals) -> list:
    """Performs one simulation step of the Replacement model"""

    ### GET THE VARIABLES NEEDED FOR THE SIMULATION STEP ###

    # Time the simulation step
    start_time = time.time()

    # Current step variables (to be reinitialized every step)
    include = ['meeting', 'position'] # options: distance, SNA metrics, meeting, position, fitness
    conclave = None #if len([node['data']['In Conclave'] for node in nodes if 'In Conclave' in node['data'] and node['data']['In Conclave'] == True]) == 0 else []
    action = None
    debug = True

    if debug:
        print('Starting Replacement Model simulation step...')

    ### GET THE VARIABLES NEEDED FOR THE SIMULATION STEP FROM THE GLOBALS ###

    # If the simulation variables are not yet initialized, initialize them
    if 'Kingpin Replacement Model' not in globals:
        globals['Kingpin Replacement Model'] = {}

    necessary_globals = {
        "Time Step (Day)": 1,
        "Removed Node IDs": [],
        "Added Node IDs": [],
        "Number of Removed Nodes": 0,
        "Number of Added Nodes": 0,
        "Run Time (s)": 0,
        "_Time Step Latest Kingpin Removed": -1,
        "_Events": [],
        "_Memory": [],
        "_New Kingpin": False,
        "_Restarted": False,
        "_Radius of Distress": 5,
        "_Current Node IDs": {}
        }
    
    for role in _potential_roles_for_replacing_role:
        necessary_globals[f"Potential Roles for Replacing {string.capwords(str(role).replace('_', ' '))}"] = (', ').join([string.capwords(str(repl_role).replace('_', ' ')) for repl_role in _potential_roles_for_replacing_role[role]])
        necessary_globals[f"_Original {string.capwords(str(role).replace('_', ' '))}"] = None

    for key, value in necessary_globals.items():
        if key not in globals['Kingpin Replacement Model']:
            globals['Kingpin Replacement Model'][key] = value

    if debug:
        print('globals intialized')

    ### GET USER PARAMETERS AND CHECK IF THE SIMULATION SHOULD BE RESTARTED OR IS FINISHED ###
        

    # Get the first variable from the globals, which is needed to check if the simulation is restarted
    restarted = True if str(globals['Kingpin Replacement Model']['_Restarted']).capitalize() == 'True' else False

    # User parameters
    automatic_intervention = params['Automatic Intervention']
    time_to_kingpin_removal = params['Time to Kingpin Removal']
    time_to_conclave = params['Time to Conclave']
    numsteps = params['Max Number of Steps']
    days_in_step = params['Number of Days in a Step']
    update_positions = params['Update Positions?']
    fix_positions = params['Fix Node Position with ID:'] if params['Fix Node Position with ID:'] != -1 else None
    restart = params['Restart Simulation?'] if not restarted else False

    if debug:
        print('gets past user parameters')

    # Check if the simulation is restarted or finished
    if restart:
        # Make sure the simulation is not restarted again in the next step
        print('The simulation is restarted.')
        restarted = True

        # Reset the globals
        for key, value in necessary_globals.items():
            globals['Kingpin Replacement Model'][key] = value

    if debug:
        print('gets past restart')
        print('hallo?')

    print('hoi')
    # Get the rest of the variables from the globals
    print(globals)
    step = int(globals['Kingpin Replacement Model']['Time Step (Day)'])
    print(step)
    timestamp_latest_kingpin_removed = int(globals['Kingpin Replacement Model']['_Time Step Latest Kingpin Removed'])
    # print(timestamp_latest_kingpin_removed)
    events = [eval(event) for event in globals['Kingpin Replacement Model']['_Events'].split(' | ')] if isinstance(globals['Kingpin Replacement Model']['_Events'], str) and len(globals['Kingpin Replacement Model']['_Events']) > 0 else globals['Kingpin Replacement Model']['_Events']
    # print(events)
    memory = [eval(memory) for memory in globals['Kingpin Replacement Model']['_Memory'].split(' | ')] if isinstance(globals['Kingpin Replacement Model']['_Memory'], str) and len(globals['Kingpin Replacement Model']['_Memory']) > 0 else globals['Kingpin Replacement Model']['_Memory']
    # print(memory)
    run_time = float(globals['Kingpin Replacement Model']['Run Time (s)'])
    # print(run_time)
    radius_of_distress = int(globals['Kingpin Replacement Model']['_Radius of Distress'])
    # print(radius_of_distress)
    current_node_ids = eval(globals['Kingpin Replacement Model']['_Current Node IDs']) if isinstance(globals['Kingpin Replacement Model']['_Current Node IDs'], str) and len(globals['Kingpin Replacement Model']['_Current Node IDs']) > 0 else globals['Kingpin Replacement Model']['_Current Node IDs']
    # print(current_node_ids)
    new_kingpin = True if str(globals['Kingpin Replacement Model']['_New Kingpin']).capitalize() == 'True' else False # seems unnecessarily complicated, could we get the boolean directly?
    # print(new_kingpin)
    if debug:
        print('globals retrieved')

    if step >= numsteps or new_kingpin == True:
        print('The simulation is finished.')
        print("Average runtime is:", round(run_time/(step - 1), 2), 's/it')
        restarted = False
        return [nodes, edges, params, globals]
    
    if debug:
        print('gets past finished check')

    # Tryout of the messages system
    # if step == 1:
    #     runcitadel.send_message(connection, f'step {step} is being simulated.', 'log')
    #     runcitadel.finalize(connection, 'success')

    if not update_positions:
        include.remove('position')
        
        if debug:
            print('positions are not updated')

    pos = {int(node['data']['id']): [node['position']['x'], node['position']['y'],node['position']['z']] for node in nodes} if 'position' in include else None

    ### INITIALIZE THE MODEL ###

    if debug:
        print('just before initializing the model')

    vn = import_vn('vn_interviews')
    model = CocaineNetwork(N = 0, t = step, vn = vn)

    events_functions = {
        'single_kingpin_removal': CocaineNetwork.event_single_kingpin_removal,
        'single_kingpin_arrest': CocaineNetwork.event_single_kingpin_arrest,
        'decide_on_new_kingpin': CocaineNetwork.event_decide_on_new_kingpin,
        'kingpin_candidate_to_kingpin_main': CocaineNetwork.event_kingpin_candidate_to_kingpin_main,
        'remove_agent_silently': CocaineNetwork.remove_agent_silently
        }
    
    # print(events_functions['decide_on_new_kingpin'])
    # for event in events_functions['decide_on_new_kingpin']:
    #     if isinstance(event[2],dict):
    #         if isinstance(event[2]['searching_replacement_for'], Node):
    #             event[2]['searching_replacement_for'] = event[2]['searching_replacement_for'].unique_id



    
    model.scheduled_events = [(t, events_functions[f], args) for t, f, args in events] if events != [''] else [] #[eval(event) for event in list(events)] if events != [''] else []

    # for t, f, args in model.scheduled_events:
    #     if f.__name__.lstrip('event_') == 'kingpin_candidate_to_kingpin_main':
    #         if isinstance(args[0], dict):
    #             if isinstance(args[0]['searching_replacement_for'],int):
    #             # search for corresponding node : 
    #                 for n in nodes:
    #                     if n.id == args[0]['searching_replacement_for'].unique_id:
    #                         print('replaced event of kingpintomain to : ',n)
    #                         args[0]['searching_replacement_for'] = n

   #for t, f, args in copy_events:
        # Replace id of old kinpin with node. 
        # if f.__name__.lstrip('event_') == 'decide_on_new_kingpin':
        #     if len(args) > 0:
        #         if isinstance(args['searching_replacement_for'], int):
        #             args['searching_replacement_for'] = model.agents[args['searching_replacement_for']]


    model.timestamp_latest_kingpin_removed = timestamp_latest_kingpin_removed
    model.memory = [(t, memory_type, id) for t, memory_type, id in memory] if memory != [''] else []

    # TODO: add a check of the graph, that all values are sensible (problem: might be slow)

    # Get the state of the graph and agents
    model.graph, model.agents = transform_Citadel_to_agents_graph(model, nodes, edges, debug=False)
    print('gets past the model translations')
    # Check the events
    # TODO: add the node transformation also for decide on new kingpin
    print(model.scheduled_events)

    for t, f, args in model.scheduled_events:
        if f.__name__ == 'event_kingpin_candidate_to_kingpin_main': # step == t and
            if isinstance(args[0], int):
                args[0] = model.agents[args[0]]
                print('replace candidate with its corresponding agent')
        # if step == t and f.__name__ == 'decide_on_new_kingpin':
        #     print('in right place to change the kingping to id ')
        #     print(args)
        #     if isinstance(args[0], int):
        #         args[0] = model.agents[args[0]]
        
        if f.__name__.lstrip('event_') == 'decide_on_new_kingpin':
            print('in second place to change the kingping to id ')
            print(t,f,args)

            if isinstance(args, dict):
                if isinstance(args['searching_replacement_for'],str):
                    # copymodel = model
                    # copymodel.agents = {id:agents for id, agents in copymodel.agents if id == args['searching_replacement_for'].id}
                    # [node],_ = transform_agents_to_Citadel_graph(copymodel, include=include, pos=pos, fixed_node_pos=fix_positions, conclave=conclave, debug=debug)
                    # args['searching_replacement_for'] = node
                    # model.graph, model.agents = transform_Citadel_to_agents_graph(model, nodes, edges, debug=False)
                    print(args['searching_replacement_for'])
                    node = eval(args['searching_replacement_for'])
                    print(node)
                    print('attributes : ')
                    print({key.lower().replace(' ','_'):val for key,val in node.items() if key in ['Financial Capital','Criminal Capital','Violence Capital']})
                    print('tags :')
                    print({key.lower().replace(' ','_'):val.lower().replace(' ','_') for key,val in node.items() if key not in ['Financial Capital','Criminal Capital','Violence Capital','Unique Id']})
                    old_id = int(node["Unique Id"])
                    #agent = Node(model=model,attributes={},tags={'business_role':'kingpin'})
                    agent = Node(unique_id=old_id,model=model,
                                 attributes={key.lower().replace(' ','_'):val for key,val in node.items() if key in ['Financial Capital','Criminal Capital','Violence Capital']},
                                 tags={key.lower().replace(' ','_'):val.lower().replace(' ','_') for key,val in node.items() if key not in ['Financial Capital','Criminal Capital','Violence Capital','Unique Id']}
                                 )
                    args['searching_replacement_for'] = agent
                    print(dir(agent))
                    pprint(vars(agent))
                    # exit()

                # search for corresponding node : 
                    # print(model.agents)
                    # print('removed agents \n')
                    # print(removed_agents)

                    #_, [agent] = transform_Citadel_to_agents_graph(model, [node], [], debug=True)
                    # args['searching_replacement_for'] = agent
                    # for id,agent in model.agents.items():
                    #     print(agent)
                    #     if id == args['searching_replacement_for']:
                    #         print('replaced event of kingpintomain to : ',agent)
                    #         args['searching_replacement_for'] = agent


            # if len(args) > 0:
            #     if isinstance(args['searching_replacement_for'], int):
            #         args['searching_replacement_for'] = model.agents[args['searching_replacement_for']]

        # CODE FROM BELOW THAT DOES THE OPPOSIT OF THAT ABOVE
        # if f.__name__.lstrip('event_') == 'decide_on_new_kingpin':
        #     if len(args) > 0:
        #         print('args not empty list')
        #         print(t,f,args)
        #         if isinstance(args['searching_replacement_for'], Node):
        #             print('args = node')
        #             print('old arg : ',args['searching_replacement_for'])
        #             print('new arg : ',args['searching_replacement_for'].unique_id)
        #             args['searching_replacement_for'] = args['searching_replacement_for'].unique_id

                            


    if debug:
        print('gets past initializing the model')

    ### HANDLE GRAPH CHANGES ###

    # Get correct current_node_ids with the role of the agents
    if isinstance(current_node_ids, (list, tuple)):
        current_node_ids = {int(node_id): model.agents[int(node_id)].role_str() for node_id in eval(globals['Kingpin Replacement Model']['_Current Node IDs']) if int(node_id) in model.agents}
        current_node_ids.update({int(node_id): 'agent' for node_id in eval(globals['Kingpin Replacement Model']['_Current Node IDs']) if int(node_id) not in model.agents})
    else:
        current_node_ids = {int(node_id): role for node_id, role in current_node_ids.items()}

    # Check if the number of nodes changed
    if int(globals['Graph Statistics']['Number of Nodes']) != len(model.graph.nodes):
        if debug:
            print('The number of nodes changed.')

        # To keep track of the changes in the graph
        removed_agents = {id: role for id, role in current_node_ids.items() if id not in model.agents}
        # removed_agents_node_instances = {id: role for id, role in current_node_ids.items() if id not in model.agents}
        for id, role in removed_agents.items():
            extra_n = 'n' if str(role[0]).lower() in ['a', 'e', 'i', 'o', 'u'] else ''
            the_or_a = 'The' if role not in set(node.role_str() for node in model.agents.values()) else f'A{extra_n}'
            role_clean = role.replace('_', ' ')
            if action != None:
                action += f'\n \t {the_or_a} {role_clean} ({id}) was removed.'
            else:
                action = f'{the_or_a} {role_clean} ({id}) was removed.'
            model.memory.append((step, 'removed', id)) # TODO: the id should be a node (question: is that even desirable?)

            # Check if the removed agent was important and should be replaced
            if role in _potential_roles_for_replacing_role:
                if len([agent.role_str() for id, agent in model.agents.items() if agent.role_str() == role]) == 0 and globals['Kingpin Replacement Model'][f"_Original {string.capwords(str(role).replace('_', ' '))}"] != None:
                    if debug:
                        print(f'The {role} was removed.')
                    
                    role_agent_id = int(eval(globals['Kingpin Replacement Model'][f"_Original {string.capwords(str(role).replace('_', ' '))}"])['Unique Id'])
                    print('Role_Agent_Id : ',role_agent_id)
                    if role_agent_id not in model.graph.nodes:
                        if role == 'kingpin':
                            model.add_event_to_schedule(step, model.event_single_kingpin_removal)
                            if (step, model.event_single_kingpin_removal, []) in model.scheduled_events:
                                model.scheduled_events.remove((step, model.event_single_kingpin_removal, []))
                        model.add_event_to_schedule(step + time_to_conclave, model.event_decide_on_new_kingpin)

                        for agent in model.agents.values(): # direct neighbors
                            if agent.distance_to_removed_kingpin is None:
                                pass  # agent is not connected at all to the removed kingpin
                            elif agent.distance_to_removed_kingpin <= radius_of_distress:
                                if agent.distance_to_removed_kingpin == 1:
                                    agent.tags[_mindset_node_tag] = _mindset_chaotic
                                    agent.tags[_activity_node_tag] = _activity_searching
                                else:
                                    agent.tags[_mindset_node_tag] = _mindset_uncertain
                            
                            node_to_replace = eval(globals['Kingpin Replacement Model'][f"_Original {string.capwords(str(role).replace('_', ' '))}"])

                            tags        = [string.capwords(str(tag).replace('_', ' ')) for tag in _node_tags_values.keys()]
                            attributes  = [string.capwords(str(attr).replace('_', ' ')) for attr in _node_attribute_values.keys()]

                            kingpin_tags_dict = {k.replace(' ', '_').lower(): v.replace(' ', '_').lower() for k, v in node_to_replace.items() if k in tags}
                            kingpin_attributes_dict = {k.replace(' ', '_').lower(): float(v) for k, v in node_to_replace.items() if k in attributes}

                            node_to_replace = Node(node_to_replace['Unique Id'], model, tags=kingpin_tags_dict, attributes=kingpin_attributes_dict)
                            agent.searching_replacement_for = node_to_replace # everyone immediately is aware of who was the removed kingpin

                        # update the globals
                        globals['Kingpin Replacement Model'][f"_Original {string.capwords(str(role).replace('_', ' '))}"] = None

                        ### record the time at which the removal occurred (may influence some behaviors of agents)
                        model.timestamp_latest_kingpin_removed = model.t

                if debug:
                    print(f'gets past handling the removal of a {role}')

    if debug:
        print('gets past handling graph changes')
    
    ### ADD EVENTS TO THE SCHEDULE ###
    # Add events to the schedule if they are not already there (if they are already there, the add_event_to_schedule function will not add them again)
    # TODO: make more robust (add to schedule)
    if step == 1 and automatic_intervention:
        model.add_event_to_schedule(time_to_kingpin_removal, model.event_single_kingpin_removal)
        model.add_event_to_schedule(time_to_kingpin_removal + time_to_conclave, model.event_decide_on_new_kingpin)

    if debug:
        print('gets past adding events to the schedule')

    ### PRINT ACTIONS THAT HAPPENED ###

    # print actions that happened before the step (between the model steps by the user)
    if action != None:
        if isinstance(action, list):
            print(action)
            print(f'step {model.t}: {action[1]}')
            if action[0]:
                new_kingpin = True
        else:
            print(f'step {model.t}: {action}')

    if debug:
        print('gets past printing actions that happened before the step')
        if step > 2:
            # print(model.agents)
            for id,agent in model.agents.items():
            #     print(agent)
            #     print(dir(agent))
                #pprint(vars(agent))
                print(agent.unique_id)
            print('gets past printing all agents before the step')
    ### MODEL STEP(s) AND MESSAGES ###
    print(model.scheduled_events)
    print(model.memory)

    # Perform "days_in_step" steps of the model
    for _ in range(days_in_step):
        # step the model
        # pprint(vars(model))
        messages = model.step()

        # print actions or messages that happened during the step
        if len(messages) > 0:
            for event, message in messages.items():
                print(event,message)
                if event[0] == 'event_single_kingpin_removal':
                    print(f'step {model.t}: {message}')
                    
                    if 'position' in include:
                        include.remove('position')
                elif event[0] == 'event_decide_on_new_kingpin':
                    if isinstance(message, list):
                        print(f'step {model.t}: {message[0]}')
                        conclave = message[1]
                        include.append('meeting')
                    else:
                        print(f'step {model.t}: {message}')
                elif event[0] == 'event_kingpin_candidate_to_kingpin_main':
                    print(f'step {model.t}: {message[1]}')
                    
                    if message[0]:
                        new_kingpin = True

                elif event[0] == 'remove_agent_silently':
                    # Sometimes the event is called, but nothing happened (the agent was already removed or not dangling)
                    if message != None:
                        print(f'step {model.t}: {message[0]}')
                else:
                    warnings.warn(f'WARNING: event "{event}" not recognized.')
        else:
            # Print just the step if nothing was already printed
            print(f'step {model.t}')

    if debug:
        print('gets past the step')


    ### UPDATE THE GLOBAL VARIABLES ###
    print('does this even happen?')
    print('newly scheduled events : ',model.scheduled_events)
    # copy_events = copy.deepcopy(model.scheduled_events)
    # TODO: make more robust
    for t, f, args in model.scheduled_events:
        if f.__name__.lstrip('event_') == 'kingpin_candidate_to_kingpin_main':
            if isinstance(args[0], Node):
                args[0] = args[0].unique_id

#    for t, f, args in copy_events:
        if f.__name__.lstrip('event_') == 'decide_on_new_kingpin':
            if len(args) > 0:
                print('args not empty list')
                print(t,f,args)
                if isinstance(args['searching_replacement_for'], Node):
                    print('args = node')
                    print('old arg : ',args['searching_replacement_for'])
                    print('new arg : ',args['searching_replacement_for'].unique_id)
                    # args['searching_replacement_for'] = args['searching_replacement_for'].unique_id
                    # str(get_attrs(role_agents[0], hidden_attributes=[attr for attr in role_agents[0].__dict__ if attr not in ['tags', 'attributes', 'unique_id']])
                    # nodes, edges = transform_agents_to_Citadel_graph(model, include=include, pos=pos, fixed_node_pos=fix_positions, conclave=conclave, debug=debug)
                    # copymodel = model
                    # copymodel.agents = {args['searching_replacement_for'].unique_id : args['searching_replacement_for']}
                    # print(copymodel.agents)
                    print(args['searching_replacement_for'])
                    # print(attr for attr in args['searching_replacement_for'].__dict__)
                    print()  #attr for attr in args['searching_replacement_for'].__dict__ if attr not in ['tags', 'attributes', 'unique_id']
                    # print(get_attrs(args['searching_replacement_for'], hidden_attributes=[attr for attr in args['searching_replacement_for'].__dict__ if attr not in ['tags', 'attributes', 'unique_id']]))

                    args['searching_replacement_for'] = str(get_attrs(args['searching_replacement_for'], hidden_attributes=[attr for attr in args['searching_replacement_for'].__dict__ if attr not in ['tags', 'attributes', 'unique_id']]))

                    # kingpinnodes,_ = transform_agents_to_Citadel_graph(copymodel, include=include, pos=pos, fixed_node_pos=fix_positions, conclave=conclave, debug=False)
                    # print(kingpinnodes)
                    # del kingpinnodes[0]['attributes']['_Searching Replacement For']
                    # # del kingpinnodes[0]['attributes']['_Agents in Conclave']
                    # # kingpinnodes[0]['attributes']['_Searching Replacement For'] = json.dumps(kingpinnodes[0]['attributes']['_Searching Replacement For'])
                    # print(kingpinnodes[0])
                    # print(str(kingpinnodes[0]))
                    # print(json.dumps(kingpinnodes[0]))
                    # args['searching_replacement_for'] = kingpinnodes[0] #json.dumps(kingpinnodes[0])
        
        # print(t,f,args)
        # # print(str(t, f.__name__.lstrip('event_'), args))
        # print()
        # print(str((t, f.__name__.lstrip('event_'), args)))

    print(model.scheduled_events)
                    
    globals['Kingpin Replacement Model']['Time Step (Day)'] = model.t
    globals['Kingpin Replacement Model']['_Time Step Latest Kingpin Removed'] = model.timestamp_latest_kingpin_removed
    globals['Kingpin Replacement Model']['_Events'] = (' | ').join([str((t, f.__name__.lstrip('event_'), args)) for t, f, args in model.scheduled_events]) if len(model.scheduled_events) > 0 else ''
    globals['Kingpin Replacement Model']['_Memory'] = (' | ').join([str((t, s, id)) for t, s, id in model.memory]) if len(model.memory) > 0 else ''
    globals['Kingpin Replacement Model']['_Restarted'] = restarted
    globals['Kingpin Replacement Model']['_New Kingpin'] = new_kingpin
    globals['Kingpin Replacement Model']['_Agents in Conclave'] = conclave

    if debug:
        print('gets past updating the first global variables')

    # Get update the original role instances
    for role in _potential_roles_for_replacing_role:
        role_agents = [agent for agent in model.agents.values() if agent.role_str() == role]
        if len(role_agents) > 0:
            globals['Kingpin Replacement Model'][f"_Original {string.capwords(str(role).replace('_', ' '))}"] =  str(get_attrs(role_agents[0], hidden_attributes=[attr for attr in role_agents[0].__dict__ if attr not in ['tags', 'attributes', 'unique_id']]))
            # globals['_Events'][]

    if debug:
        print('gets past updating original global variables')

    if len(memory) > 0 and memory != ['']:
        for memory_type in _memory_types:
            if memory_type == 'removed':
                globals['Kingpin Replacement Model']['Removed Node IDs'] = [id for time, type, id in memory if type == memory_type]
            elif memory_type == 'added':
                globals['Kingpin Replacement Model']['Added Node IDs'] = [id for time, type, id in memory if type == memory_type]
            else:
                warnings.warn('WARNING: memory type not recognized.')

    if debug:
        print('gets past updating memory')

    globals['Kingpin Replacement Model']['Number of Removed Nodes'] = len(globals['Kingpin Replacement Model']['Removed Node IDs'])
    globals['Kingpin Replacement Model']['Number of Added Nodes'] = len(globals['Kingpin Replacement Model']['Added Node IDs'])
    globals['Kingpin Replacement Model']['_Current Node IDs'] = str({str(id): node.role_str() for id, node in model.agents.items()})

    if debug:
        print('gets past updating the kingpin replacement model variables')

    # Recompute globals variables which are not stored in the model
    globals['Graph Statistics'] = check_graph_statistics(graph=model.graph)
    
    if debug:
        print('gets past updating all global variables')

    ### TRANSFORM THE AGENTS BACK TO A GRAPH ###

    if len(model.graph.nodes) == 0 or len(model.graph.edges) == 0:
        warnings.warn('WARNING: the graph is empty.')
        #terminate('The graph is empty.')
        return [None, None, None, None]

    nodes, edges = transform_agents_to_Citadel_graph(model, include=include, pos=pos, fixed_node_pos=fix_positions, conclave=conclave, debug=False)

    if debug:
        print('gets past transforming the agents back to a graph')

    end_time = time.time()
    globals['Kingpin Replacement Model']['Run Time (s)'] = run_time + (end_time - start_time)

    # traceback.print_stack()
    print(globals)

    return [nodes, edges, params, globals]

# Parameters that can be set by the user
# {'name': 'Kingpin Removal Time', 'type': 'slider', 'min': 0, 'max': 100, 'step': 1, 'value': 0},
# {'name': 'Kingpin Decide Time', 'type': 'slider', 'min': 0, 'max': 100, 'step': 1, 'value': 5},
startParams = [{
                'attribute': 'Automatic Intervention',
                'type': 'boolean',
                'defaultValue': True,
                'value': True,
                "limits": None
            },
            {
                'attribute': 'Time to Kingpin Removal',
                'type': 'integer',
                'defaultValue': 4,
                'value': 4,
                'limits': {
                    'min': 1,
                    'max': 60
                }
            },
            {   
                'attribute': 'Time to Conclave',
                'type': 'integer',
                'defaultValue': 5,
                'value': 5,
                'limits': {
                    'min': 1,
                    'max': 60
                }
            },
            {
                'attribute': 'Max Number of Steps',
                'type': 'integer',
                'defaultValue': 100,
                'value': 100,
                'limits': {
                    'min': 1,
                    'max': 1000
                }
            },
            {
                'attribute': 'Number of Days in a Step',
                'type': 'integer',
                'defaultValue': 1,
                'value': 1,
                'limits': {
                    'min': 1,
                    'max': 7
                }
            },
            {
                'attribute': 'Update Positions?',
                'type': 'boolean',
                'defaultValue': False,
                'value': False,
                'limits': None
            },
            {
                'attribute': 'Restart Simulation?',
                'type': 'boolean',
                'defaultValue': False,
                'value': False,
                'limits': None
            },
            {
                'attribute': 'Fix Node Position with ID:',
                'type': 'integer',
                'defaultValue': -1,
                'value': -1,
                'limits': {
                    'min': 0,
                    'max': 0
                }
            }]

# Main
if __name__ == "__main__":
    # Assert required information is provided
    if (len(sys.argv) != 5):
        # print("Usage: python kingpin_simstep.py url port sid key")

        # Ask again
        print("Please provide the \"Args\" from Citadel to connect to the simulator.")
        terminal_input = input("url, port, sid, and key: ").split(" ")

        if (len(terminal_input) != 4):
            print("Usage: url port sid key. You provided: " + str(terminal_input))
            exit(1)

        url = terminal_input[0]
        port = int(terminal_input[1])
        sid = terminal_input[2]
        key = terminal_input[3]
    else:
        # Take these from the command line
        url = sys.argv[1]
        port = int(sys.argv[2])
        sid = sys.argv[3]
        key = sys.argv[4]
    
    title = 'Criminal Cocaine Replacement Model'
    
    # Connect
    asyncio.run(runcitadel.connect(url, port, sid, key, title, startParams, ccrm_sim_step))