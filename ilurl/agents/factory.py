from typing import ClassVar
from importlib import import_module

_IMPORT_ERROR = (" Unrecognized agent's type. "
                "Error while importing module `{0}`. "
                "Make sure the agent's implementation is "
                "correctly placed in the respective module. "
                "As an example, the implementation of the `DQN` "
                "agent must be placed in the following module: "
                "ilurl.agents.dqn.agent (pay attention to the "
                "fact that `dqn` is lowercased in the path).  "
                "Moreover, the agent's class defined in the "
                "module ilurl.agents.dqn.agent module must be "
                "named `DQN`.")


class AgentFactory(object):
    """
        This class imports and retrieves a reference to a class
        that implements a given agent's type.

        Class attributes:
        ----------------
        * _class = reference to agent's class.

        Class methods:
        -------------
        * get: imports and returns a reference to a given agent's type.

    """
    _class = None

    @classmethod
    def get(cls, name : str) -> ClassVar:

        print(name)

        if not cls._class:

            module_path = f'ilurl.agents.{name.lower()}.agent'
            try:
                agent_class = getattr(import_module(module_path), name)
            except Exception as e:
                print(e)
                raise ModuleNotFoundError(_IMPORT_ERROR.format(module_path))

            cls._class = agent_class

            return agent_class

        else:
            return cls._class
