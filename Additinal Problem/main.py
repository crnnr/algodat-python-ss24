import numpy as np
import pandas as pd
import pydotplus
from graphviz import Source
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

def tree_to_dot(tree, graph=None, parent_name=None, edge_label=""):
    """
    Convert the tree dictionary to DOT format for visualization.
    """
    if graph is None:
        graph = pydotplus.Dot(graph_type='digraph')
    
    if isinstance(tree, dict):
        for attribute, branches in tree.items():
            node_name = f"attribute_{attribute}"
            node = pydotplus.Node(node_name, label=f"Attribute {attribute}")
            graph.add_node(node)
            if parent_name:
                edge = pydotplus.Edge(parent_name, node_name, label=edge_label)
                graph.add_edge(edge)
            for value, subtree in branches.items():
                tree_to_dot(subtree, graph, node_name, f"Value {value}")
    else:
        leaf_name = f"class_{tree}"
        leaf = pydotplus.Node(leaf_name, label=f"Class {tree}", shape='box')
        graph.add_node(leaf)
        if parent_name:
            edge = pydotplus.Edge(parent_name, leaf_name, label=edge_label)
            graph.add_edge(edge)
    
    return graph

# Example usage
if __name__ == "__main__":
    # Dataset from 12.5 because why not
    data = [
        ['sunny', 'warm', 'normal', 'strong', 'warm', 'same', '+'],
        ['sunny', 'warm', 'high', 'strong', 'warm', 'same', '+'],
        ['rainy', 'cold', 'high', 'strong', 'warm', 'change', '-'],
        ['sunny', 'warm', 'high', 'strong', 'cool', 'change', '+'],
        ['sunny', 'warm', 'normal', 'weak', 'warm', 'same', '-']
    ]
    
    # Convert to pandas DataFrame for easier handling
    columns = ['sky', 'air', 'humid', 'wind', 'water', 'forecast', 'attack']
    df = pd.DataFrame(data, columns=columns)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])
    
    # Convert back to list of lists
    examples = df.values.tolist()
    
    # Separate features and labels for sklearn DecisionTreeClassifier
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Train the decision tree
    clf = DecisionTreeClassifier(criterion='entropy')  # You can use 'gini' or 'entropy'
    clf.fit(X, y)
    
    # Visualize the sklearn tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=columns[:-1], class_names=le.classes_)
    plt.show()
    
    # Custom decision tree
    attributes = list(range(X.shape[1]))  # Indices of the attributes in examples
    tree = dt_learning(examples, attributes)
    print(tree)
    
    # Visualize the custom tree
    dot_graph = tree_to_dot(tree)
    dot_graph_str = dot_graph.to_string()
    print("Generated DOT format:\n", dot_graph_str)  # Debugging: print the generated DOT format

    dot_graph.write("decision_tree.dot", format='dot')
    graphviz_source = Source.from_file("decision_tree.dot")
    graphviz_source.render("decision_tree", format='png', cleanup=True)
