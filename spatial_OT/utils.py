from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.stats import entropy
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import scanpy as sc
from scipy.spatial.distance import jensenshannon


def build_knn_graph_nocelltype(df, k=3):
    # Initialize graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in range(len(df)):
        G.add_node(i, spatial_position=(df.loc[i, 'x'], df.loc[i, 'y']))
    
    # Use k-NN to find neighbors
    spatial_positions = df[['x', 'y']].values
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(spatial_positions)  # k+1 because the node itself will be included
    distances, indices = nbrs.kneighbors(spatial_positions)
    
    # Add edges based on k-NN
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip the first neighbor, which is the node itself
            G.add_edge(i, neighbor)
    return G


def build_knn_graph(df, k=3):
    """
    Build a k-Nearest Neighbors (k-NN) graph from the given DataFrame.

    Parameters:
        df (DataFrame): A pandas DataFrame with 'spatial_position' and 'cell_type' columns.
        k (int): The number of neighbors to connect each node to.

    Returns:
        G (Graph): A NetworkX graph with nodes and edges based on k-NN.
    """

    # Initialize an empty graph
    G = nx.Graph()
    
    # Add nodes to the graph with spatial positions and cell types as attributes
    for i in range(len(df)):
        G.add_node(
            i, 
            spatial_position=df.loc[i, 'spatial_position'], 
            cell_type=df.loc[i, 'cell_type']
        )
    
    # Extract spatial positions (assumes they are in a column named 'spatial_position')
    # Converts them into a numpy array for efficient distance computations
    spatial_positions = df[['spatial_position']].values  
    
    # Use k-NN to find neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(spatial_positions)  # k+1 to include the node itself
    distances, indices = nbrs.kneighbors(spatial_positions)  # Compute distances and indices of neighbors
    
    # Add edges to the graph based on k-NN relationships
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip the first neighbor (the node itself)
            G.add_edge(i, neighbor)  # Create an edge between the current node and its neighbor
    
    return G  # Return the constructed k-NN graph

def build_knn_graph_from2d(df, k=3):
    # Initialize graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in range(len(df)):
        G.add_node(i, spatial_position=(df.loc[i, 'x'], df.loc[i, 'y']), cell_type=df.loc[i, 'cell_type'])
    
    # Use k-NN to find neighbors
    spatial_positions = df[['x', 'y']].values
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(spatial_positions)  # k+1 because the node itself will be included
    distances, indices = nbrs.kneighbors(spatial_positions)
    
    # Add edges based on k-NN
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip the first neighbor, which is the node itself
            G.add_edge(i, neighbor)
    return G

def build_knn_graph_expression(df, cell_types, k=3):
    # Initialize graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in range(len(df)):
        G.add_node(i, cell_type=cell_types[i])
    
    # Use k-NN to find neighbors
    expression_data = df.values
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(expression_data)  # k+1 because the node itself will be included
    distances, indices = nbrs.kneighbors(expression_data)
    
    # Add edges based on k-NN
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip the first neighbor, which is the node itself
            G.add_edge(i, neighbor)
    return G

def build_knn_graph_proportions(df, stgtcelltype, k=3):
    # Initialize graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in range(len(df)):
        G.add_node(i, spatial_position=(df.loc[i, 'x'], df.loc[i, 'y']), cell_type_proportions = stgtcelltype.iloc[i])
    
    # Use k-NN to find neighbors
    spatial_positions = df[['x', 'y']].values
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(spatial_positions)  # k+1 because the node itself will be included
    distances, indices = nbrs.kneighbors(spatial_positions)
    
    # Add edges based on k-NN
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip the first neighbor, which is the node itself
            G.add_edge(i, neighbor)
    return G

def get_neighbor_distribution(graph):
    """
    Compute the distribution of neighbor cell types for each node in a graph.
    
    Parameters:
        graph (networkx.Graph): Input graph where each node has a 'cell_type' attribute.

    Returns:
        dict: A dictionary where keys are node IDs and values are dictionaries
              representing the cell type distribution of neighbors.
    """
    distributions = {}
    
    for node in graph.nodes:
        # Get neighbors' cell types
        neighbors = list(graph.neighbors(node))
        neighbor_cell_types = [graph.nodes[neighbor]['cell_type'] for neighbor in neighbors]

        if not neighbor_cell_types:
            distributions[node] = {}
            continue

        # Count cell types and normalize to get probabilities
        cell_type_counts = Counter(neighbor_cell_types)
        total_neighbors = sum(cell_type_counts.values())
        distributions[node] = {cell_type: count / total_neighbors for cell_type, count in cell_type_counts.items()}
    
    return distributions

def compute_kl_matrix(graph1, graph2):
    """
    Compute a KL divergence matrix between nodes of two graphs based on their neighbor distributions.

    Parameters:
        graph1 (networkx.Graph): First graph.
        graph2 (networkx.Graph): Second graph.

    Returns:
        np.ndarray: A matrix where the entry (i, j) represents the KL divergence
                    between node i in graph1 and node j in graph2.
    """
    dist1 = get_neighbor_distribution(graph1)
    dist2 = get_neighbor_distribution(graph2)

    all_cell_types = set()
    for d in dist1.values():
        all_cell_types.update(d.keys())
    for d in dist2.values():
        all_cell_types.update(d.keys())
    all_cell_types = sorted(all_cell_types)

    # Helper function to convert a distribution to a fixed vector
    def to_vector(distribution):
        return np.array([distribution.get(cell_type, 0.0) for cell_type in all_cell_types])

    # Compute KL divergence matrix
    kl_matrix = np.zeros((len(graph1.nodes), len(graph2.nodes)))

    for i, node1 in enumerate(graph1.nodes):
        p = to_vector(dist1[node1]) + 1e-10  # Add small value to avoid log(0)
        for j, node2 in enumerate(graph2.nodes):
            q = to_vector(dist2[node2]) + 1e-10
            kl_matrix[i, j] = entropy(p, q)

    return kl_matrix

def process_transport_dataframe(T, index_labels, column_labels):
    """
    Process a transport DataFrame to group by the provided index and column labels, 
    normalize by row sums, and return the processed DataFrame.
    
    Parameters:
        T (pd.DataFrame): The transport matrix.
        index_labels (list or array-like): Labels for the rows of the DataFrame.
        column_labels (list or array-like): Labels for the columns of the DataFrame.
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Assign the provided index and column labels
    T_df = pd.DataFrame(T)
    T_df.index = index_labels
    T_df.columns = column_labels
    
    # Group by columns and sum
    T_df = T_df.groupby(by=T_df.columns, axis=1).sum()
    
    # Group by rows and sum
    T_df = T_df.groupby(by=T_df.index, axis=0).sum()
    
    # Normalize rows by their sums
    T_df = T_df.div(T_df.sum(axis=1), axis=0) * 100
    
    return T_df

def compute_spatial_entropy(graph):
    spatial_entropies = {}
    
    # For each node in the graph
    for node in graph.nodes:
        # Get the cell type of the current node
        current_cell_type = graph.nodes[node]['cell_type']
        
        
        # Get neighbors' cell types
        neighbor_cell_types = [graph.nodes[neighbor]['cell_type'] for neighbor in graph.neighbors(node)]
        
        if not neighbor_cell_types:
            spatial_entropies[node] = 0  # No neighbors, entropy is zero
            continue
        
        # Compute frequency of each cell type in neighbors
        cell_type_counts = pd.Series(neighbor_cell_types).value_counts()
        probabilities = cell_type_counts / cell_type_counts.sum()
        
        # Compute entropy
        entropy = -np.sum(probabilities * np.log(probabilities))
        spatial_entropies[node] = entropy
    
    return spatial_entropies

def compute_spatial_entropy_with_proportions(graph):
    """
    Compute spatial entropy for each node based on neighbors' cell type proportions.

    Parameters:
        graph (networkx.Graph): A graph where nodes have cell type proportions as attributes.

    Returns:
        dict: A dictionary mapping each node to its spatial entropy.
    """
    spatial_entropies = {}

    # Iterate through each node in the graph
    for node in graph.nodes:
        # Get neighbors
        neighbors = list(graph.neighbors(node))
        
        if not neighbors:
            spatial_entropies[node] = 0.0  # No neighbors, entropy is zero
            continue

        # Aggregate cell type proportions from neighbors
        neighbor_proportions = pd.DataFrame(
            [pd.Series(graph.nodes[neighbor]['cell_type_proportions']) for neighbor in neighbors]
        ).sum(axis=0)

        # Normalize proportions to sum to 1
        probabilities = neighbor_proportions / neighbor_proportions.sum()

        # Compute entropy
        entropy = -np.sum(probabilities * np.log(probabilities))
        spatial_entropies[node] = entropy

    return spatial_entropies

def compute_average_neighbor_expression(graph, expression_matrix):
    """
    Compute the average expression of neighbors for each node and return as a matrix.
    
    Parameters:
        graph (networkx.Graph): A graph where nodes represent cells or samples.
        expression_matrix (pandas.DataFrame): A DataFrame with gene expression values,
                                              where rows correspond to node indices 
                                              and columns correspond to genes.
    
    Returns:
        pandas.DataFrame: A DataFrame where each row corresponds to a node and contains
                          the average expression of its neighbors.
    """
    # Initialize a matrix to store average expressions
    average_expression_matrix = []

    # Iterate through each node in the graph
    for node in graph.nodes:
        # Get the neighbors of the current node
        neighbors = list(graph.neighbors(node))
        
        if not neighbors:
            # If no neighbors, return a zero vector (or NaN vector if preferred)
            avg_expression = pd.Series(0, index=expression_matrix.columns)
        else:
            # Compute the average expression across neighbors
            avg_expression = expression_matrix.iloc[neighbors].mean(axis=0)
        
        # Append the average expression vector
        average_expression_matrix.append(avg_expression.values)

    # Convert the list to a DataFrame
    average_expression_matrix = pd.DataFrame(
        average_expression_matrix,
        columns=expression_matrix.columns,
        index=expression_matrix.index
    )
    
    return average_expression_matrix

def intersect(lst1, lst2): 
    """
    param: lst1 - list
    param: lst2 - list
    
    return: list of common elements
    """
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3 

def compute_accuracy_max_prob(transport_matrix, source_labels, target_labels):

    source_labels = np.asarray(source_labels)
    target_labels = np.asarray(target_labels)

    # Get the target cell index with max probability for each source cell
    max_prob_indices = np.argmax(transport_matrix, axis=1)
    
    # Predicted cell types based on max probability
    predicted_labels = target_labels[max_prob_indices]
    # Compare predicted labels with source labels
    correct_predictions = (source_labels == predicted_labels)
    
    # Compute accuracy
    accuracy = correct_predictions.mean()
    
    return accuracy


def assign_coordinates(transport_map, spatial_coords):
    """
    Assign x, y coordinates to single cells based on the transport map.

    Parameters:
        transport_map (ndarray): 2D array (n_cells x n_spots) where rows are single cells and columns are spatial spots.
        spatial_coords (pd.DataFrame): DataFrame with columns ['x', 'y'], representing spatial spot coordinates.

    Returns:
        pd.DataFrame: DataFrame with assigned x, y coordinates for each single cell.
    """
    # Normalize transport map rows to ensure they sum to 1 (probabilities)
    transport_map = transport_map / transport_map.sum(axis=1, keepdims=True)
    
    # Compute weighted average coordinates for each single cell
    x_coords = np.dot(transport_map, spatial_coords['x'].values)
    y_coords = np.dot(transport_map, spatial_coords['y'].values)
    
    # Create DataFrame for single cells with assigned coordinates
    single_cell_coords = pd.DataFrame({'x': x_coords, 'y': y_coords})
    return single_cell_coords

def compute_mae(predicted_coords, true_coords):
    """
    Compute the Mean Absolute Error (MAE) between predicted and true coordinates.

    Parameters:
        predicted_coords (ndarray): Predicted coordinates, shape (n_cells, 2).
        true_coords (ndarray): True coordinates, shape (n_cells, 2).

    Returns:
        float: Mean Absolute Error (MAE) between predicted and true coordinates.
    """
    mae = np.mean(np.sum(np.abs(predicted_coords - true_coords), axis=1))
    return mae

def rotate_coordinates(coords, angle_degrees):
    """
    Rotate 2D coordinates by a given angle.

    Parameters:
        coords (ndarray): Original coordinates (n x 2).
        angle_degrees (float): Angle to rotate the coordinates (in degrees).

    Returns:
        ndarray: Rotated coordinates (n x 2).
    """
    angle_radians = np.radians(angle_degrees)  # Convert angle to radians
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    return np.dot(coords, rotation_matrix.T)


def plot_multiple_cell_types(spot_coords, cell_type_proportions, cell_types, title="Cell Type Distribution"):
    """
    Plot the distribution of multiple cell types on a Visium map.

    Parameters:
        spot_coords (ndarray): Spatial coordinates (n_spots x 2).
        cell_type_proportions (pd.DataFrame): DataFrame with cell type proportions.
        cell_types (list): List of cell types to plot.
        title (str): Title for the overall plot.
    """
    n_types = len(cell_types)
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 6), squeeze=False)

    rotated_coords = rotate_coordinates(spot_coords, 180)
    
    # Plot each cell type
    for i, cell_type in enumerate(cell_types):
        if cell_type not in cell_type_proportions.columns:
            raise ValueError(f"Cell type '{cell_type}' not found in the proportions DataFrame.")
        
        scatter = axes[0, i].scatter(
            rotated_coords[:, 0], rotated_coords[:, 1],
            c=cell_type_proportions[cell_type], cmap="coolwarm", s=100, edgecolor="black"
        )
        fig.colorbar(scatter, ax=axes[0, i], label=f"Proportion of {cell_type}")
        axes[0, i].set_title(f"{cell_type} Distribution")
        axes[0, i].set_xlabel("X Coordinate")
        axes[0, i].set_ylabel("Y Coordinate")
        axes[0, i].set_aspect('equal', adjustable='box')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def compute_spot_cell_type_proportions(transport_map, cell_types):
    """
    Normalize the transport map and compute cell type proportions for each Visium spot.

    Parameters:
        transport_map (ndarray): Transport map (rows=single cells, columns=spots).
        cell_types (list or ndarray): List of cell types for each single cell.

    Returns:
        pd.DataFrame: DataFrame where rows are spots and columns are cell type proportions.
    """
    # Normalize transport map columns (spots) to sum to 1
    normalized_transport_map = transport_map / transport_map.sum(axis=1, keepdims=True)
    
    # Create a DataFrame for cell types
    cell_type_df = pd.DataFrame({
        'cell_type': cell_types,
        'cell_index': np.arange(len(cell_types))
    })

    # Compute weighted cell type proportions for each spot
    spot_cell_type_contributions = []
    for spot_idx in range(normalized_transport_map.shape[1]):
        # Transport probabilities for all cells to this spot
        weights = normalized_transport_map[:, spot_idx]
        
        # Weighted count of cell types
        weighted_counts = cell_type_df.groupby('cell_type').apply(
            lambda group: np.sum(weights[group['cell_index']])
        )
        
        # Normalize to get proportions
        proportions = weighted_counts / weighted_counts.sum()
        spot_cell_type_contributions.append(proportions)

    # Convert to DataFrame
    spot_cell_type_df = pd.DataFrame(spot_cell_type_contributions, index=[f"Spot_{i}" for i in range(transport_map.shape[1])])
    return spot_cell_type_df

def compute_cell_type_colocalization(coords, cell_type_proportions, k=5):
    """
    Compute cell type co-localization across Visium spots.

    Parameters:
        coords (ndarray): Spatial coordinates of Visium spots (n_spots x 2).
        cell_type_proportions (pd.DataFrame): DataFrame of cell type proportions (spots x cell types).
        k (int): Number of nearest neighbors to consider.

    Returns:
        pd.DataFrame: Co-localization matrix for cell types.
    """
    # Compute k-NN for the Visium spots
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)  # k+1 because the spot itself is included
    _, indices = nbrs.kneighbors(coords)  # Indices of nearest neighbors

    # Compute neighbor-enriched proportions
    neighbor_enriched_proportions = []
    for spot_idx in range(coords.shape[0]):
        # Get neighbors (exclude the spot itself)
        neighbors = indices[spot_idx, 1:]
        # Compute average proportions across neighbors
        neighbor_proportion = cell_type_proportions.iloc[neighbors].mean()
        neighbor_enriched_proportions.append(neighbor_proportion)

    # Convert to DataFrame
    neighbor_enriched_proportions = pd.DataFrame(neighbor_enriched_proportions, index=cell_type_proportions.index)

    # Compute co-localization matrix (correlation between cell types)
    colocalization_matrix = neighbor_enriched_proportions.corr()

    return colocalization_matrix


def mapping_accuracy(labels1, labels2, pi):
    """
    Compute mapping accuracy using transport probabilities for string labels.

    Parameters:
    - labels1: np.ndarray, shape (n_source,)
        NumPy array of **string** labels for the first spatial slice.
    - labels2: np.ndarray, shape (n_target,)
        NumPy array of **string** labels for the second spatial slice.
    - pi: np.ndarray, shape (n_source, n_target)
        Transport matrix where pi[i, j] represents the probability of mapping 
        spot i in slice 1 to spot j in slice 2.

    Returns:
    - accuracy: float
        The computed mapping accuracy (normalized between 0 and 1).
    """

    # Define mapping dictionary for categorical labels
    unique_labels = np.unique(np.concatenate([labels1, labels2]))
    mapping_dict = {label: i for i, label in enumerate(unique_labels)}

    # Convert string labels to numerical values using the mapping dictionary
    labels1_mapped = pd.Series(labels1).map(mapping_dict).to_numpy()
    labels2_mapped = pd.Series(labels2).map(mapping_dict).to_numpy()

    # Compute binary match matrix (1 if same label, 0 otherwise)
    label_match = labels1_mapped[:, None] == labels2_mapped[None, :]

    # Compute weighted accuracy
    weighted_accuracy = np.sum(pi * label_match) 

    # Normalize by total transport mass
    accuracy = weighted_accuracy / np.sum(pi)

    return accuracy


def compute_brier_score(T, source_labels, target_labels):
    """
    Compute Brier Score to assess transport probabilities.
    
    Parameters:
    - T: np.ndarray, shape (n_source, n_target)
        Transport probability matrix where T[i, j] is the probability 
        of source cell i being transported to target cell j.
    - source_labels: np.ndarray, shape (n_source,)
        Ground-truth cell type labels of the source cells.
    - target_labels: np.ndarray, shape (n_target,)
        Ground-truth cell type labels of the target cells.

    Returns:
    - brier_score: float
        Brier Score for the transport predictions.
    """
    T = T / T.sum(axis=1, keepdims=True) 

    # Convert labels to string (or categorical type) to avoid unhashable numpy arrays
    source_labels = np.array([str(label) for label in source_labels])
    target_labels = np.array([str(label) for label in target_labels])

    # Create mapping of unique cell types to numerical indices
    unique_labels = np.unique(np.concatenate([source_labels, target_labels]))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Convert target labels to one-hot encoding
    target_onehot = np.zeros((len(target_labels), len(unique_labels)))
    for i, label in enumerate(target_labels):
        target_onehot[i, label_to_index[label]] = 1

    # Compute probability distributions for each source cell
    predicted_probs = np.dot(T, target_onehot)

    # Convert source labels to one-hot encoding
    source_onehot = np.zeros_like(predicted_probs)
    for i, label in enumerate(source_labels):
        if label in label_to_index:
            source_onehot[i, label_to_index[label]] = 1

    # Compute Brier Score
    brier_score = np.mean((predicted_probs - source_onehot) ** 2)

    return brier_score


def compute_transported_adata_argmax(adata_source, adata_target, transport_matrix):
    """
    Compute transported AnnData object using the argmax method.

    Parameters:
        adata_source (AnnData): Original AnnData before transport.
        adata_target (AnnData): Target AnnData after transport.
        transport_matrix (ndarray): Transport matrix (N_source x N_target).

    Returns:
        AnnData: New AnnData with updated spatial coordinates.
    """
    # Extract spatial coordinates
    spatial_target = adata_target.obsm["spatial"]

    # Find the target index with the highest probability for each source cell
    max_indices = np.argmax(transport_matrix, axis=1)  # Shape: (N_source,)

    # Assign new spatial positions from the most probable target cell
    spatial_transported = spatial_target[max_indices]  # Shape: (N_source, 2)

    # Create a new AnnData object for transported data
    adata_transported = adata_source.copy()
    adata_transported.obsm["spatial"] = spatial_transported  # Update spatial positions

    return adata_transported

def compute_local_cell_type_distribution(adata, k=10, cell_type_key="cell_type"):
    """
    Compute the local cell-type distribution for each cell.

    Parameters:
        adata (AnnData): AnnData object with spatial coordinates and cell types.
        k (int): Number of nearest neighbors to consider.

    Returns:
        ndarray: Local cell-type distributions (N_cells x N_cell_types).
    """
    # Extract spatial coordinates and cell types
    spatial_coords = adata.obsm["spatial"]
    cell_types = adata.obs[cell_type_key].astype("category").cat.codes.values  # Convert categorical to numerical
    unique_types = np.unique(cell_types)  # Get unique cell type indices
    num_types = len(unique_types)  # Number of unique cell types

    # Compute k-nearest neighbors
    knn = NearestNeighbors(n_neighbors=k).fit(spatial_coords)
    neighbors = knn.kneighbors(spatial_coords, return_distance=False)

    # Compute local cell-type distribution
    local_distributions = np.zeros((spatial_coords.shape[0], num_types))

    for i in range(spatial_coords.shape[0]):
        neighbor_types = cell_types[neighbors[i]]  # Get types of neighbors
        for j, cell_type in enumerate(unique_types):
            local_distributions[i, j] = np.sum(neighbor_types == cell_type) / k  # Normalize

    return local_distributions

def compute_js_divergence_before_after(adata_source, adata_transported, k=5, cell_type_key="cell_type"):
    """
    Compute the average Jensen-Shannon (JS) divergence between the local cell-type distributions
    before and after transport.

    Parameters:
        adata_source (AnnData): Original AnnData before transport.
        adata_transported (AnnData): AnnData after transport.
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: Average JS divergence across all cells.
    """
    # Compute local cell-type distributions
    source_distributions = compute_local_cell_type_distribution(adata_source, k, cell_type_key)
    transported_distributions = compute_local_cell_type_distribution(adata_transported, k, cell_type_key)

    # Compute JS divergence for each cell
    js_divergences = [
        jensenshannon(source_distributions[i], transported_distributions[i])
        for i in range(source_distributions.shape[0])
    ]

    # Return average JS divergence across all cells
    return np.mean(js_divergences)
