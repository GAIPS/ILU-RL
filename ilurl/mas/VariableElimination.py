import copy
import json
import random
from itertools import product
from ilurl.mas.CGAgent import CGAgent
from ilurl.mas.ActionTable import ActionTable


def maximizeAgent(agent, action_dict):
    _max = ("-1", -1)
    # Figure out the max and maxArg of current agent actions
    for agent_action in agent.possible_actions:
        _sum = 0
        actions = dict({agent.name: agent_action}, **action_dict)
        # Maximizing the sum of every payout function
        for function in agent.payout_functions:
            _sum += function.get_value(actions)
        if _sum >= _max[1]:
            _max = (agent_action, _sum)
    return _max


def variable_elimination(agents, debug=False, epsilon=0, test=False):
    elimination_agents = list(agents.values())

    # First Pass
    for agent in elimination_agents:

        # For every agent that depends on current agent
        dependant_agents = [agents[agent_name] for agent_name in agent.dependant_agents]

        # If last agent to eliminate, create best response.
        if len(dependant_agents) == 0:
            agent.best_response = ActionTable([])
            _max = maximizeAgent(agent, {})

            if not test and random.random() < epsilon:
                action = random.choice(agent.possible_actions)  # Select random action
                agent.best_response.set_action(action)
            else:
                agent.best_response.set_action(_max[0])
            continue

        action_product = list(product(*[dependant_agent.possible_actions for dependant_agent in dependant_agents]))

        new_function = ActionTable(dependant_agents)
        agent.best_response = ActionTable(dependant_agents)

        # For every action pair of dependant agents
        for joint_action in action_product:
            action_dict = {agent.dependant_agents[i]: joint_action[i] for i in range(len(agent.dependant_agents))}

            _max = maximizeAgent(agent, action_dict)

            # Save new payout and best response
            if not test and random.random() < epsilon:
                action = random.choice(agent.possible_actions)  # Select random action
                agent.best_response.set_value(action_dict, action)
                _sum = 0
                for function in agent.payout_functions:  # Get value for new action
                    _sum += function.get_value(
                        dict({agent.name: random.choice([n for n in agent.possible_actions if n != _max[0]])},
                             **action_dict))

                new_function.set_value(action_dict, _sum)

            else:
                agent.best_response.set_value(action_dict, _max[0])
                new_function.set_value(action_dict, _max[1])

        # Delete all payout functions that involve the parent agent from all the dependant agents
        # And add the new payout functions to dependants
        for dependant_agent in agent.dependant_agents:
            # Remove all functions that have agent_name in the dependants
            agents[dependant_agent].payout_functions = [function for function in
                                                        agents[dependant_agent].payout_functions if
                                                        agent.name not in function.agent_names]
            if agent.name in agents[dependant_agent].dependant_agents:
                agents[dependant_agent].dependant_agents.remove(agent.name)

            agents[dependant_agent].payout_functions.append(new_function)

            # Add all dependants (except himself) to the agent's list if they are not already in
            agents[dependant_agent].dependant_agents.extend([agent_name for agent_name in agent.dependant_agents if
                                                             agent_name != dependant_agent and agent_name not in agents[
                                                                 dependant_agent].dependant_agents])

    # Second Pass, Reverse Order
    actions = {}
    for agent in list(elimination_agents)[::-1]:
        actions[agent.name] = int(agent.best_response.get_value(actions))

    return actions
