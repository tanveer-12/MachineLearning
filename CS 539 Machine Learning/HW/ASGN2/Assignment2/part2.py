import pandas as pd
from part1 import Tree  # Import your decision tree implementation

#-----------------------------------------------
# Helper function to encode categorical variables
def encode_categorical(data):
    """
    Encode categorical variables in the dataset using label encoding.
    :param data: The dataset (pandas DataFrame).
    :return: The encoded dataset (pandas DataFrame) and a dictionary of label encoders.
    """
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':  # Check if the column is categorical
            unique_values = data[column].unique()
            mapping = {value: idx for idx, value in enumerate(unique_values)}  # Create a mapping
            data[column] = data[column].map(mapping)  # Apply the mapping
            label_encoders[column] = mapping  # Save the encoder for later use
    return data, label_encoders

#-----------------------------------------------
# Load the dataset
def load_dataset(filename):
    """
    Load the dataset from a file.
    :param filename: The name of the file (string).
    :return: The feature matrix (X) and target labels (Y) as numpy arrays.
    """
    # Load the dataset using pandas
    data = pd.read_csv(filename, delimiter='\t')
    
    # Encode categorical variables
    data, label_encoders = encode_categorical(data)
    
    # Split features and target
    X = data.drop('Risk', axis=1).values.T  # Transpose to match the format expected by part1.py
    Y = data['Risk'].values
    
    return X, Y, label_encoders

#-----------------------------------------------
# Function to print the decision tree in a readable format
def print_tree(node, feature_names, label_mapping, depth=0):
    """
    Recursively print the decision tree in a readable format.
    :param node: The current node in the decision tree.
    :param feature_names: The names of the features (list).
    :param label_mapping: The mapping of encoded labels to original labels (dict).
    :param depth: The current depth of the tree (int).
    """
    if node.isleaf:
        # If the node is a leaf, print the predicted label
        print("  " * depth + "Predict:", label_mapping[node.p])
        return
    
    # Print the current attribute and its values
    attribute_name = feature_names[node.i]
    print("  " * depth + f"{attribute_name} = ?")
    
    # Recursively print the children nodes
    for value, child_node in node.C.items():
        print("  " * (depth + 1) + f"| {attribute_name} = {value}")
        print_tree(child_node, feature_names, label_mapping, depth + 2)

#-----------------------------------------------
# Function to identify unused features in the decision tree
def identify_unused_features(tree, feature_names):
    """
    Identify features that do not play a role in the decision tree.
    :param tree: The root of the decision tree (Node object).
    :param feature_names: The names of the features (list).
    :return: A list of unused feature names.
    """
    used_features = set()  # Set to store features used in the tree

    # Recursively traverse the tree to find used features
    def traverse(node):
        if node.isleaf:
            return
        used_features.add(feature_names[node.i])  # Add the current feature
        for child_node in node.C.values():
            traverse(child_node)  # Recursively traverse child nodes

    traverse(tree)  # Start traversal from the root

    # Find unused features
    unused_features = [feature for feature in feature_names if feature not in used_features]
    return unused_features

#-----------------------------------------------
# Predict credit risk for new instances
def predict_new_instances(tree, new_data, label_encoders):
    """
    Predict the credit risk for new instances using the trained decision tree.
    :param tree: The trained decision tree (Tree object).
    :param new_data: The new instances (pandas DataFrame).
    :param label_encoders: The dictionary of label encoders.
    :return: The predicted credit risk for the new instances (list).
    """
    # Encode the new data using the same label encoders
    for column in new_data.columns:
        if column in label_encoders:
            new_data[column] = new_data[column].map(label_encoders[column])
    
    # Convert the new data to the format expected by part1.py
    X_new = new_data.values.T  # Transpose to match the format expected by part1.py
    
    # Predict the credit risk
    predictions = Tree.predict(tree, X_new)  # Use the predict method from the Tree class
    
    # Reverse the label encoding for the predictions
    risk_mapping = {v: k for k, v in label_encoders['Risk'].items()}  # Reverse the Risk mapping
    predictions = [risk_mapping[pred] for pred in predictions]  # Map predictions back to original labels
    
    return predictions

#-----------------------------------------------
# Main function for Part 2
def main():
    # Load the dataset
    X, Y, label_encoders = load_dataset('credit.txt')
    
    # Build the decision tree
    tree = Tree.train(X, Y)  # Use the train method from the Tree class to build the tree
    
    # Print the decision tree in a readable format
    feature_names = ["Debt", "Income", "Married?", "Owns_Property", "Gender"]  # Replace with actual feature names
    risk_mapping = {v: k for k, v in label_encoders['Risk'].items()}  # Reverse the Risk mapping
    print("Decision Tree:")
    print_tree(tree, feature_names, risk_mapping)
    
    # Predict credit risk for Tom and Ana
    new_data = pd.DataFrame({
        'Debt': ['low', 'low'],
        'Income': ['low', 'medium'],
        'Married?': ['no', 'yes'],
        'Owns_Property': ['no', 'yes'],
        'Gender': ['male', 'female']
    }, index=['Tom', 'Ana'])
    
    predictions = predict_new_instances(tree, new_data, label_encoders)
    print("\nPredicted Credit Risk:")
    for name, pred in zip(new_data.index, predictions):
        print(f"{name}: {pred}")
    
    # Identify unused features in the decision tree
    unused_features = identify_unused_features(tree, feature_names)
    print("\nUnused Features in Decision Tree:", unused_features)

#-----------------------------------------------
# Run the main function
if __name__ == "__main__":
    main()