import copy
import json
import random
from itertools import product
from ilurl.mas.CGAgent import CGAgent
from ilurl.mas.ActionTable import ActionTable

def variable_elimination(agents, debug=False, epsilon=0, test=False):


    elimination_agents = list(agents.values())

        # First Pass
    for agent in elimination_agents:

        # For every agent that depends on current agent
        dependant_agent_names = agent.dependant_agents
        dependant_agents = [agents[agent_name] for agent_name in dependant_agent_names]
        # Create all action possibilities between those agents
        # action_product = product(*[agent.possible_actions for agent in dependant_agents])

        if len(dependant_agents) == 0:
            _max = ("-1", -1)
            agent.best_response = ActionTable([])
            for agent_action in agent.possible_actions:
                _sum = 0
                actions = {agent.name: agent_action}
                # Maximizing the sum of every local payout function
                for function in agent.payout_functions:
                    _sum += function.get_value(actions)
                if _sum >= _max[1]:
                    _max = (agent_action, _sum)

            if not test and random.random() < epsilon:
                # action = random.choice([n for n in agent.possible_actions if n != _max[0]]) # Select random suboptimal action
                action = random.choice(agent.possible_actions) # Select random suboptimal action

                agent.best_response.set_action(action)
            else:
                agent.best_response.set_action(_max[0])
            continue

        action_product = list(product(*[dependant_agent.possible_actions for dependant_agent in dependant_agents]))

        new_function = ActionTable(dependant_agents)
        agent.best_response = ActionTable(dependant_agents)

        # For every action pair of dependant agents
        for joint_action in action_product:
            _max = ("-1", -1)
            action_dict = {dependant_agent_names[i]: joint_action[i] for i in range(len(dependant_agent_names))}
            # Figure out the max and maxArg of current agent actions
            for agent_action in agent.possible_actions:
                _sum = 0
                actions = dict({agent.name: agent_action}, **action_dict)
                # Maximizing the sum of every local payout function
                for function in agent.payout_functions:
                    #TODO: Get Q Values from ACME HERE
                    _sum += function.get_value(actions)
                if _sum >= _max[1]:
                    _max = (agent_action, _sum)

            # Save new payout and best response
            if not test and random.random() < epsilon:
                #action = random.choice([n for n in agent.possible_actions if n != _max[0]]) # Select random suboptimal action
                action = random.choice(agent.possible_actions) # Select random suboptimal action

                agent.best_response.set_value(action_dict, action)
                _sum = 0
                for function in agent.payout_functions: # Get value for new action
                    _sum += function.get_value(
                        dict({agent.name: random.choice([n for n in agent.possible_actions if n != _max[0]])},
                             **action_dict))

                new_function.set_value(action_dict, _sum)

            else:
                agent.best_response.set_value(action_dict, _max[0])
                new_function.set_value(action_dict, _max[1])

        # Delete all payout functions that involve the parent agent from all the dependant agents
        # And add the new payout functions to dependants
        for agent_name in dependant_agent_names:
            # Remove all functions that have agent_name in the dependants
            agents[agent_name].payout_functions = [function for function in agents[agent_name].payout_functions if
                                                   agent.name not in function.agent_names]
            if agent.name in agents[agent_name].dependant_agents:
                agents[agent_name].dependant_agents.remove(agent.name)

            agents[agent_name].payout_functions.append(new_function)

            # Add all dependants (except himself) to the agent's list if they are not already in
            agents[agent_name].dependant_agents.extend([agent for agent in dependant_agent_names if
                                                        agent != agent_name and agent not in agents[
                                                            agent_name].dependant_agents])

    # Second Pass, Reverse Order, excluding the last agent
    # last_agent = list(elimination_agents)[-1]
    # actions = {last_agent.name: str(last_agent.payout_functions[0].table.argmax().data[()])}

    for agent in list(elimination_agents)[::-1]:
        actions[agent.name] = int(agent.best_response.get_value(actions))
    if debug:
        print("\nVariable Elimination Result:")
        for key, value in sorted(actions.items(), key=lambda x: x[0]):
            print("{} : {}".format(key, value), end=', ')
    # cleanup(agents)

    return actions

def cleanup(agents):
    for agent in agents.values():
        agent.payout_functions = [agent.qtable]
