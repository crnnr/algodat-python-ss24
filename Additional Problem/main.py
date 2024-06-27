import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import LabelEncoder

def plurality_val(examples):
    """
    Returns the most common output value among a set of examples.
    """
    values, counts = np.unique([example[-1] for example in examples], return_counts=True)
    return values[np.argmax(counts)]

def importance(attribute, examples):
    """
    Calculate the importance of an attribute (Information Gain).
    """
    def entropy(examples):
        values, counts = np.unique([example[-1] for example in examples], return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))
    
    entropy_before = entropy(examples)
    values, counts = np.unique([example[attribute] for example in examples], return_counts=True)
    weighted_entropy_after = sum((counts[i] / counts.sum()) * entropy([example for example in examples if example[attribute] == values[i]]) for i in range(len(values)))
    
    return entropy_before - weighted_entropy_after

def dt_learning(examples, attributes, parent_examples=None):
    """
    Decision tree learning algorithm.
    """
    if examples == []:
        return plurality_val(parent_examples)
    elif all(example[-1] == examples[0][-1] for example in examples):
        return examples[0][-1]
    elif not attributes:
        return plurality_val(examples)
    else:
        best_attribute = max(attributes, key=lambda attr: importance(attr, examples))
        tree = {best_attribute: {}}
        for value in [0, 1]:  # Binary attributes assumed to be 0 or 1
            exs = [example for example in examples if example[best_attribute] == value]
            subtree = dt_learning(exs, [attr for attr in attributes if attr != best_attribute], examples)
            tree[best_attribute][value] = subtree
        return tree

def plot_tree(tree, feature_names, class_names):
    """
    Plot the decision tree using networkx and matplotlib.
    """
    def add_edges(tree, graph, parent=None, label=''):
        if isinstance(tree, dict):
            for attribute, branches in tree.items():
                node_label = feature_names[attribute]
                node_name = f"{node_label}\n(Attribute {attribute})"
                graph.add_node(node_name)
                if parent:
                    graph.add_edge(parent, node_name, label=label)
                for value, subtree in branches.items():
                    add_edges(subtree, graph, node_name, str(value))
        else:
            class_name = class_names[tree]
            leaf_name = f"Class: {class_name}"
            graph.add_node(leaf_name)
            if parent:
                graph.add_edge(parent, leaf_name, label=label)

    graph = nx.DiGraph()
    add_edges(tree, graph)

    pos = hierarchy_pos(graph, next(iter(graph.nodes)))
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    plt.show()

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    If the graph is a tree, this will return the positions to plot this in a hierarchical layout.
    """
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)  
    
    if len(children) != 0:
        dx = width / len(children) 
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    
    return pos

# Example usage
if __name__ == "__main__":
    # Dataset
    data = [
        ['high', 'none', 'normal', 'low', 'weak', 'False'],
        ['low', 'strong', 'high', 'middle', 'strong', 'True'],
        ['middle', 'none', 'high', 'middle', 'weak', 'True'],
        ['low', 'strong', 'high', 'strong', 'strong',  'True'],
        ['low', 'strong', 'normal', 'high', 'weak', 'True'],
        ['middle', 'none', 'high', 'middle', 'weak', 'True'],
        ['high', 'none', 'normal', 'low', 'weak', 'False'],
        ['middle', 'none', 'high', 'middle', 'weak', 'True'],
        ['high', 'strong', 'normal', 'high', 'weak', 'True'],
        ['low', 'strong', 'normal', 'middle', 'weak', 'True'],
        ['high', 'none', 'normal', 'middle', 'strong', 'False'],
        ['middle', 'strong', 'high', 'middle', 'strong', 'True'],
        ['low', 'none', 'normal', 'middle', 'weak', 'False'],
        ['high', 'none', 'high', 'high', 'weak', 'False'],
        ['low', 'none', 'normal', 'low', 'weak', 'False'],
        ['middle', 'none', 'high', 'middle', 'weak', 'True'],
        ['low', 'strong', 'high', 'strong', 'strong', 'True'],
        ['high', 'weak', 'normal', 'low', 'weak', 'False'],
    ]
    
    # Convert to pandas DataFrame for easier handling
    columns = ['temperature', 'wind', 'probability_rain', 'humid', 'water', 'take_umbrella']
    df = pd.DataFrame(data, columns=columns)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])
    
    # Convert back to list of lists
    examples = df.values.tolist()
    
    # Custom decision tree
    attributes = list(range(df.shape[1] - 1))  # Indices of the attributes in examples
    tree = dt_learning(examples, attributes)
    print(tree)
    
    # Plot the custom tree
    feature_names = columns[:-1]
    class_names = le.classes_
    plot_tree(tree, feature_names, class_names)
