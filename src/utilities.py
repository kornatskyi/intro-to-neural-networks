from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(value):
        if value not in nodes:
            nodes.add(value)
            for child in value._prev:
                edges.add((child, value))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "TB"})  # LR = left to right

    nodes, edges = trace(root)
    for node in nodes:
        uid = str(id(node))
        # for any value in a grpahql create rectangular node for it
        dot.node(
            name=uid,
            label="{ %s | data: %.4f | grad: %.4f }"
            % (node.label, node.data, node.grad),
            shape="record",
        )
        if node._op:
            dot.node(name=uid + node._op, label=node._op)
            dot.edge(uid + node._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
