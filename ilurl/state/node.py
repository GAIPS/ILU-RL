def get_instance_name(x):
    """Gets the name of the instance"""
    return x.__class__.__name__.lower()

def is_unique(xs):
    """Tests if all x in xs belong to the same instance"""
    fn = get_instance_name
    xs_set = {x for x in map(fn, xs)}
    if len(xs_set) == 1:
        return list(xs_set)[0]
    return False

class Node:
    """Node into the state tree hierarchy

      * Provides a means for bi-direction communication
        between parent and children.
      * Provides bfs function.
      * Implements sequence protocol.
      * Thin-wrapper around domain functionality.
    """
    def __init__(self, parent, node_id, children):
        self.parent = parent
        self.node_id = node_id
        self.children = children
        # Creates an alias
        if children is not None and any(children):
            alias = is_unique(children.values())
            if alias:
                setattr(self, f'{alias}s', children)

    # Sequence protocol
    def __len__(self):
        return len(self.children)

    def __getitem__(self, index):
        return self.children[index]

    # DFS for paths
    def search_path(self, node_id, path, seek_root=True):
        """Returns a path ending on node_ID"""
        # 1) Base case: this is the element
        if node_id == self.node_id:
            return True
        else:
            # 2) Seek root node first.
            if seek_root:
                if self.parent is not None:
                    self.parent.search_path(node_id, path)

            found = False
            # 3) Search down strem
            for chid, ch in self.children.items():
                path.append(chid)
                found = ch.search_path(node_id, path, seek_root=False)
                if found:
                    break
                del path[-1]
            return found



