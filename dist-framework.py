#!/usr/bin/env python3
# By Thibaut LOMBARD (@lombardweb)
# dist-framework.py
# a Framework for testing and discovering distances, and metrics 
# and exploring Maths Formulas
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cosine, cityblock, hamming, minkowski, mahalanobis
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import skew, kurtosis
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score, mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import ripser
from persim import plot_diagrams, PersImage
import ot
import sympy as sp
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

# --- Step 1: Load documents and prepare data ---
documents_folder = "documents"
if not os.path.exists(documents_folder):
    os.makedirs(documents_folder)
    print(f"Created folder '{documents_folder}'. Please add .txt files to this folder and rerun the script.")
    exit()

txt_files = [f for f in os.listdir(documents_folder) if f.endswith('.txt')]
num_docs = len(txt_files)

if num_docs == 0:
    print(f"No .txt files found in the '{documents_folder}' folder. Please add .txt files and rerun the script.")
    exit()

print(f"Found {num_docs} documents in the '{documents_folder}' folder.")

# Read documents and build a vocabulary
documents = []
doc_labels = []
all_terms = set()

for i, file_name in enumerate(txt_files):
    with open(os.path.join(documents_folder, file_name), 'r', encoding='utf-8') as f:
        content = f.read().lower()
        documents.append(content)
        doc_labels.append(file_name.replace('.txt', ''))
        terms = re.findall(r'\b\w+\b', content)
        all_terms.update(terms)

vocabulary = sorted(list(all_terms))
num_terms = len(vocabulary)
print(f"Total unique terms in vocabulary: {num_terms}")

# Represent documents as vectors (term frequencies)
term_to_index = {term: idx for idx, term in enumerate(vocabulary)}
doc_vectors = np.zeros((num_docs, num_terms))
for i, doc in enumerate(documents):
    terms_in_doc = re.findall(r'\b\w+\b', doc.lower())
    for term in terms_in_doc:
        if term in term_to_index:
            doc_vectors[i, term_to_index[term]] += 1

# Simulate document labels for mutual information
np.random.seed(42)
true_labels = np.random.randint(0, 3, size=num_docs)

# --- Step 2: Apply Transformations and Embeddings ---
# Normalize the vectors (scale and units)
scaler = StandardScaler()
doc_vectors_normalized = scaler.fit_transform(doc_vectors)

# Define embedding methods
embedding_methods = {
    "PCA": lambda X: PCA(n_components=min(5, num_terms, num_docs)).fit_transform(X),
    "t-SNE": lambda X: TSNE(n_components=min(2, num_terms, num_docs), random_state=42).fit_transform(X),
    "UMAP": lambda X: umap.UMAP(n_components=min(2, num_terms, num_docs), random_state=42).fit_transform(X)
}

# Prompt user to select embedding method
print("\nAvailable Embedding Methods:")
for i, method in enumerate(embedding_methods.keys(), 1):
    print(f"{i}. {method}")
while True:
    try:
        choice = int(input("Select an embedding method (enter the number): ")) - 1
        if 0 <= choice < len(embedding_methods):
            selected_embedding = list(embedding_methods.keys())[choice]
            break
        else:
            print(f"Please enter a number between 1 and {len(embedding_methods)}.")
    except ValueError:
        print("Please enter a valid number.")

print(f"\nSelected embedding method: {selected_embedding}")

# Apply the selected embedding
doc_vectors_transformed = embedding_methods[selected_embedding](doc_vectors_normalized)
print(f"Transformed document vectors to {doc_vectors_transformed.shape[1]} dimensions using {selected_embedding}.")

# --- Step 3: Define Available Distance Metrics ---
all_metrics = {
    "Euclidean": {"implemented": True, "description": "Straight-line distance (L2 norm)", "space": "Vector Space"},
    "Manhattan": {"implemented": True, "description": "Sum of absolute differences (L1 norm)", "space": "Vector Space"},
    "Chebyshev": {"implemented": True, "description": "Maximum of absolute differences (L∞ norm)", "space": "Vector Space"},
    "Minkowski": {"implemented": True, "description": "Generalized distance with p=3", "space": "Vector Space"},
    "Mahalanobis": {"implemented": True, "description": "Distance accounting for covariance", "space": "Vector Space"},
    "Cosine": {"implemented": True, "description": "Cosine of the angle between vectors", "space": "Vector Space"},
    "Hamming": {"implemented": True, "description": "Number of differing positions", "space": "String Space"},
    "Levenshtein": {"implemented": True, "description": "Minimum edit distance", "space": "String Space"},
    "KL Divergence": {"implemented": True, "description": "Divergence between distributions", "space": "Probability Space"},
    "Jensen-Shannon": {"implemented": True, "description": "Symmetrized KL divergence", "space": "Probability Space"},
    "Bhattacharyya": {"implemented": True, "description": "Similarity between distributions", "space": "Probability Space"},
    "Hellinger": {"implemented": True, "description": "Bounded similarity between distributions", "space": "Probability Space"},
    "Wasserstein": {"implemented": True, "description": "Optimal transport distance", "space": "Probability Space"},
    "Alcubierre": {"implemented": True, "description": "Warped distance inspired by Alcubierre metric", "space": "Vector Space"},
    "Persistent Homology": {"implemented": True, "description": "Topological distance using persistent homology", "space": "Topological Space"},
    "Neural Network": {"implemented": True, "description": "Learned distance using a neural network", "space": "Vector Space"},
    "Fibonacci": {"implemented": True, "description": "Distance based on Fibonacci progression", "space": "Sequence Space"},
}

# --- Step 4: User Prompt for Selecting Metrics ---
print("\nAvailable Distance Metrics:")
for i, (metric, info) in enumerate(all_metrics.items(), 1):
    print(f"{i}. {metric} ({info['space']}): {info['description']}")

selected_metrics = []
while True:
    try:
        choices = input("\nSelect the metrics to use (enter numbers separated by commas, e.g., 1,2,3), or type 'all' to select all implemented metrics: ").strip()
        if choices.lower() == 'all':
            selected_metrics = [metric for metric, info in all_metrics.items() if info["implemented"]]
            break
        else:
            choices = [int(x.strip()) - 1 for x in choices.split(',')]
            for choice in choices:
                metric = list(all_metrics.keys())[choice]
                if not all_metrics[metric]["implemented"]:
                    print(f"Warning: {metric} is not implemented and will be skipped.")
                else:
                    selected_metrics.append(metric)
            if selected_metrics:
                break
            else:
                print("No valid implemented metrics selected. Please try again.")
    except (ValueError, IndexError):
        print("Invalid input. Please enter numbers corresponding to the metrics, separated by commas, or 'all'.")

print(f"\nSelected Metrics: {selected_metrics}")

# Load operators table
operators_df = pd.read_csv("operators_table.csv")
languages = operators_df.columns[1:-2]
operator_to_languages = {}
operator_to_tex = {}
for _, row in operators_df.iterrows():
    operator = row['Operator']
    operator_to_languages[operator] = []
    for lang in languages:
        value = str(row[lang]).strip()
        if value.startswith("Yes"):
            operator_to_languages[operator].append(lang)
    operator_to_tex[operator] = row['TeX']

# --- Step 5: Neural Network for Learning Custom Distance Metrics ---
class DistanceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(DistanceNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
    
    def forward(self, x):
        return self.encoder(x)

def train_distance_net(vectors, epochs=100):
    input_dim = vectors.shape[1]
    model = DistanceNet(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    triplet_loss = nn.TripletMarginLoss(margin=1.0)
    
    # Create triplets (anchor, positive, negative)
    vectors_tensor = torch.FloatTensor(vectors)
    labels = torch.LongTensor(true_labels)
    triplets = []
    for i in range(len(vectors)):
        anchor = vectors_tensor[i]
        pos_idx = random.choice([idx for idx, label in enumerate(labels) if label == labels[i] and idx != i])
        neg_idx = random.choice([idx for idx, label in enumerate(labels) if label != labels[i]])
        positive = vectors_tensor[pos_idx]
        negative = vectors_tensor[neg_idx]
        triplets.append((anchor, positive, negative))
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for anchor, positive, negative in triplets:
            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(triplets):.4f}")
    
    return model

def compute_neural_distance(vectors, model):
    model.eval()
    with torch.no_grad():
        vectors_tensor = torch.FloatTensor(vectors)
        embeddings = model(vectors_tensor).numpy()
    dist_matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            dist_matrix[i, j] = euclidean(embeddings[i], embeddings[j])
    return dist_matrix

# Train the neural network if selected
neural_model = None
if "Neural Network" in selected_metrics:
    print("\nTraining neural network for custom distance metric...")
    neural_model = train_distance_net(doc_vectors_transformed)

# --- Step 6: Fibonacci-Specific Distance Metric ---
def fibonacci_distance(u, v):
    """
    Return 0 if u and v follow Fibonacci progression, else a penalty based on deviation.
    Assumes u and v are sequences of numbers extracted from documents.
    """
    try:
        # Convert document content to sequences of numbers
        u_nums = [float(x) for x in u if str(x).replace('.', '', 1).isdigit()]
        v_nums = [float(x) for x in v if str(x).replace('.', '', 1).isdigit()]
        
        if len(u_nums) < 3 or len(v_nums) < 3:
            return float('inf')  # Not enough numbers to check Fibonacci property
        
        # Check if sequences follow Fibonacci rule: F(n+2) = F(n) + F(n+1)
        def is_fibonacci_sequence(nums):
            for i in range(len(nums) - 2):
                if not np.isclose(nums[i + 2], nums[i] + nums[i + 1], rtol=1e-2):
                    return False
            return True
        
        # Check for inverse Fibonacci: 1/F(n)
        def is_inverse_fibonacci(nums):
            inverses = [1/x for x in nums if x != 0]
            return is_fibonacci_sequence(inverses)
        
        # Check for Fibonacci ratios: F(n+1)/F(n) approaching golden ratio
        def is_fibonacci_ratio(nums):
            ratios = [nums[i+1]/nums[i] for i in range(len(nums)-1) if nums[i] != 0]
            golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618
            return all(np.isclose(r, golden_ratio, rtol=0.1) for r in ratios[2:])  # Skip first few ratios
        
        # Compute deviation from Fibonacci properties
        u_fib = is_fibonacci_sequence(u_nums)
        v_fib = is_fibonacci_sequence(v_nums)
        u_inv_fib = is_inverse_fibonacci(u_nums)
        v_inv_fib = is_inverse_fibonacci(v_nums)
        u_ratio = is_fibonacci_ratio(u_nums)
        v_ratio = is_fibonacci_ratio(v_nums)
        
        if (u_fib and v_fib) or (u_inv_fib and v_inv_fib) or (u_ratio and v_ratio):
            return 0  # Both sequences follow a Fibonacci-related pattern
        else:
            # Compute a penalty based on deviation
            min_len = min(len(u_nums), len(v_nums))
            diff = np.abs(np.array(u_nums[:min_len]) - np.array(v_nums[:min_len]))
            return np.sum(diff)
    except:
        return float('inf')  # Return high penalty if conversion fails

# --- Step 7: Metric Validity Check ---
def is_metric(dist_matrix, tol=1e-8):
    """
    Checks if a given distance matrix is a metric:
    - Non-negativity
    - Symmetry
    - Identity of indiscernibles
    - Triangle inequality
    """
    n = dist_matrix.shape[0]

    # 1. Non-negativity
    if np.any(dist_matrix < -tol):
        return False, "Non-negativity violated"

    # 2. Identity of indiscernibles: d(i,i) == 0
    if not np.allclose(np.diag(dist_matrix), 0, atol=tol):
        return False, "Identity violated"

    # 3. Symmetry: d(i,j) == d(j,i)
    if not np.allclose(dist_matrix, dist_matrix.T, atol=tol):
        return False, "Symmetry violated"

    # 4. Triangle inequality: d(i,k) ≤ d(i,j) + d(j,k)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if dist_matrix[i, k] > dist_matrix[i, j] + dist_matrix[j, k] + tol:
                    return False, f"Triangle inequality violated at ({i},{j},{k})"

    return True, "Valid metric"

# --- Step 8: Operator Fuzzing for New Distance Formulas ---
def fuzz_operators(operators, num_combinations=10):
    operator_list = operators['Operator'].tolist()
    fuzzed_formulas = []
    for _ in range(num_combinations):
        num_ops = random.randint(1, 3)
        selected_ops = random.sample(operator_list, num_ops)
        formula = "d1"
        for op in selected_ops:
            if op in ['&&', '||', '>', '<', '>=', '<=', '==', '!=']:
                continue
            formula += f" {op} d2"
        fuzzed_formulas.append((formula, selected_ops))
    return fuzzed_formulas

def discover_new_distance_formulas(operators_df, operator_to_tex, existing_distances, num_fuzzed=10):
    new_distances = {}
    distance_names = list(existing_distances.keys())
    fuzzed_formulas = fuzz_operators(operators_df, num_fuzzed)
    
    for i, dist1_name in enumerate(distance_names):
        for dist2_name in distance_names[i+1:]:
            for formula, ops in fuzzed_formulas:
                new_dist_name = f"{dist1_name}_{'_'.join(ops)}_{dist2_name}"
                try:
                    tex_ops = [operator_to_tex.get(op, op) for op in ops]
                    tex_desc = f"d_{{ \\text{{{new_dist_name}}}}} (u, v) = d_{{ \\text{{{dist1_name}}}}} (u, v) {' '.join(tex_ops)} d_{{ \\text{{{dist2_name}}}}}"
                    
                    def new_dist_func(u, v, dist1=existing_distances[dist1_name], dist2=existing_distances[dist2_name], operators=ops):
                        d1 = dist1[u, v]
                        d2 = dist2[u, v]
                        result = d1
                        for op in operators:
                            if op == '+':
                                result += d2
                            elif op == '-':
                                result = np.abs(result - d2)
                            elif op == '*':
                                result *= d2
                            elif op == '/':
                                result /= (d2 + 1e-10)
                            elif op == '^':
                                result = result ** d2 if d2 >= 0 else 0
                        return result
                    
                    new_distances[new_dist_name] = {
                        "function": new_dist_func,
                        "tex": tex_desc
                    }
                except Exception as e:
                    print(f"Error creating new distance {new_dist_name}: {e}")
    
    return new_distances

# Parse and validate formulas
def parse_formula_sympy(formula, operator_to_langs, known_vars):
    tokens = re.split(r'(\W+)', formula)
    tokens = [t.strip() for t in tokens if t.strip()]
    identified_operators = []
    operator_languages = {}
    variables = []
    unidentified = []
    
    for token in tokens:
        if token in operator_to_langs:
            identified_operators.append(token)
            operator_languages[token] = operator_to_langs[token]
        elif token in known_vars:
            variables.append(token)
        elif token.isidentifier() or token.replace('.', '', 1).isdigit():
            variables.append(token)
        else:
            unidentified.append(token)
    
    return identified_operators, operator_languages, variables, unidentified

def validate_and_evaluate_formula(formula, data, var_mapping, unidentified_values, feature_names, operator_to_languages, name="Formula"):
    try:
        identified_operators, _, variables, unidentified = parse_formula_sympy(formula, operator_to_languages, feature_names)
        if not identified_operators:
            raise ValueError(f"{name} does not contain any recognized operators.")
        
        symbols = list(var_mapping.keys()) + [var for var in unidentified_values.keys() if isinstance(var, str)]
        sympy_vars = {var: sp.Symbol(var) for var in symbols}
        
        expression = formula.replace('&&', 'and').replace('||', 'or').replace('^', '**')
        expression = expression.replace('==', '=').replace('!=', '!=')
        
        for var, idx in var_mapping.items():
            expression = expression.replace(var, str(data[idx]))
        for unid, value in unidentified_values.items():
            if isinstance(value, (int, float)):
                expression = expression.replace(unid, str(value))
            else:
                expression = expression.replace(unid, str(value))
        
        expr = sp.sympify(expression, locals=sympy_vars)
        results = []
        for doc_idx in range(len(data)):
            subs_dict = {sympy_vars[var]: data[doc_idx] for var in var_mapping.keys()}
            for unid, value in unidentified_values.items():
                if isinstance(value, (int, float)):
                    subs_dict[sympy_vars[unid]] = value
                else:
                    subs_dict[sympy_vars[unid]] = value[doc_idx]
            result = float(expr.subs(subs_dict))
            results.append(result)
        return np.array(results)
    except Exception as e:
        print(f"Error validating/evaluating {name}: {e}")
        return None

def formula_to_tex(formula, operator_to_tex, feature_names):
    tokens = re.split(r'(\W+)', formula)
    tokens = [t.strip() for t in tokens if t.strip()]
    tex_parts = []
    for token in tokens:
        if token in operator_to_tex:
            tex_parts.append(operator_to_tex[token])
        elif token in feature_names or token.isidentifier() or token.replace('.', '', 1).isdigit():
            tex_parts.append(f"\\text{{{token}}}")
        else:
            tex_parts.append(token)
    return "".join(tex_parts)

# Threshold formula: x_j > R * (1 - theta)
R = 1.5  # Real number parameter (can be adjusted)
def apply_threshold(j, theta, R):
    return j > R * (1 - theta)

# Univariate assessment with state-of-the-art metrics
def univariate_assessment(data, name):
    mean = np.mean(data)
    variance = np.var(data)
    skewness = skew(data)
    kurt = kurtosis(data)
    data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-10)
    shannon_entropy = -np.sum(data_normalized * np.log(data_normalized + 1e-10))
    return {
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurt,
        "shannon_entropy": shannon_entropy,
        f"{name}_values": data
    }

# Multivariate assessment
def multivariate_assessment(j_data, theta_data):
    covariance = np.cov(j_data, theta_data)[0, 1]
    correlation = np.corrcoef(j_data, theta_data)[0, 1]
    j_discrete = np.digitize(j_data, bins=np.linspace(j_data.min(), j_data.max(), 5))
    theta_discrete = np.digitize(theta_data, bins=np.linspace(theta_data.min(), theta_data.max(), 5))
    mutual_info = mutual_info_score(j_discrete, theta_discrete)
    return {
        "covariance": covariance,
        "correlation": correlation,
        "mutual_info": mutual_info
    }

# Commutation of j and theta
def commute_j_theta(j_data, theta_data):
    return j_data * theta_data

# Visualization functions
def plot_heatmap(dist_matrix, title, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.show()

def plot_dendrogram(dist_matrix, title, labels, method):
    linked = linkage(dist_matrix, method=method)
    plt.figure(figsize=(8, 6))
    dendrogram(linked, labels=labels, orientation='top')
    plt.title(f"{title} (Linkage: {method})")
    plt.show()

def plot_entropy(entropies, title, labels, selected=None):
    plt.figure(figsize=(8, 6))
    colors = ['skyblue' if selected is None or i in selected else 'gray' for i in range(len(labels))]
    plt.bar(labels, entropies, color=colors)
    plt.title(title)
    plt.xlabel("Documents")
    plt.ylabel("Entropy")
    plt.show()

def plot_metrics_usage(selected_metrics, all_metrics, config_name):
    plt.figure(figsize=(12, 6))
    metrics = list(all_metrics.keys())
    status = [1 if metric in selected_metrics else 0.5 if all_metrics[metric]["implemented"] else 0 for metric in metrics]
    colors = ['green' if s == 1 else 'blue' if s == 0.5 else 'red' for s in status]
    plt.bar(metrics, status, color=colors)
    plt.title(f"Metrics Usage in {config_name}\n(Green: Used, Blue: Implemented but Not Used, Red: Not Implemented)")
    plt.xlabel("Metrics")
    plt.ylabel("Status")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"metrics_usage_{config_name.replace(' ', '_')}.png")
    plt.close()

# Compute evaluation metrics
def compute_clustering_metrics(dist_matrix, true_labels):
    if np.all(dist_matrix == 0):
        return -1, float('inf'), 0
    clustering = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average')
    cluster_labels = clustering.fit_predict(dist_matrix)
    if len(np.unique(cluster_labels)) < 2:
        return -1, float('inf'), 0
    
    silhouette = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
    try:
        davies_bouldin = davies_bouldin_score(dist_matrix, cluster_labels)
    except:
        davies_bouldin = float('inf')  # Handle edge cases where DB score fails
    mutual_info = mutual_info_score(true_labels, cluster_labels)
    return silhouette, davies_bouldin, mutual_info

def compute_entropy_metrics(entropy, true_labels):
    variance = np.var(entropy)
    entropy_discrete = np.digitize(entropy, bins=np.linspace(entropy.min(), entropy.max(), 5))
    mutual_info = mutual_info_score(true_labels, entropy_discrete)
    return variance, mutual_info

def compute_composite_score(silhouette, davies_bouldin, mutual_info, is_entropy=False):
    norm_silhouette = (silhouette + 1) / 2
    norm_davies_bouldin = 1 / (1 + davies_bouldin)
    norm_mutual_info = np.tanh(mutual_info)
    
    if is_entropy:
        return 0.5 * norm_mutual_info + 0.5 * np.tanh(variance)
    return 0.4 * norm_silhouette + 0.3 * norm_davies_bouldin + 0.3 * norm_mutual_info

# Alcubierre-inspired metric
def alcubierre_dist(u, v, warp_factor=1.0):
    euclid = np.sqrt(np.sum((u - v) ** 2))
    warped = euclid * np.tanh(warp_factor * euclid)
    return warped

# Persistent Homology Distance
def compute_persistent_homology_distance(vectors):
    diagrams = []
    for vec in vectors:
        point_cloud = vec.reshape(-1, 1)
        diagram = ripser.ripser(point_cloud)['dgms']
        diagrams.append(diagram)
    
    dist_matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i == j:
                dist_matrix[i, j] = 0
                continue
            dist = persim.wasserstein(diagrams[i][0], diagrams[j][0])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

# Compute theta as a distance-based measure
def compute_theta_as_distance(doc_vectors_transformed, best_divergence, divergences, doc_idx):
    """
    Compute theta as the average distance from doc_idx to all other documents using the best divergence metric.
    """
    dist_matrix = divergences[best_divergence]
    theta = np.mean(dist_matrix[doc_idx])  # Average distance to all other documents
    return theta

# Compute properties for the main formula
def compute_formula_properties(x_j, theta, doc_vectors_transformed, selected_embedding, j):
    """
    Compute norms, inner products, transformations, scale/units, and geometry/topology for the main formula.
    """
    properties = {}

    # Ensure x_j and theta are arrays for consistent computation
    x_j = np.asarray(x_j)
    theta = np.asarray(theta)

    # Norms for x_j
    properties["L1_norm_xj"] = np.linalg.norm(x_j, ord=1)
    properties["L2_norm_xj"] = np.linalg.norm(x_j, ord=2)
    properties["Linf_norm_xj"] = np.linalg.norm(x_j, ord=np.inf)

    # Compute xj_minus_theta
    xj_minus_theta = x_j - theta
    properties["L1_norm_xj_minus_theta"] = np.linalg.norm(xj_minus_theta, ord=1)
    properties["L2_norm_xj_minus_theta"] = np.linalg.norm(xj_minus_theta, ord=2)
    properties["Linf_norm_xj_minus_theta"] = np.linalg.norm(xj_minus_theta, ord=np.inf)

    # Inner Products
    # Compute inner product; ensure scalar output
    if x_j.ndim == 0 and theta.ndim == 0:  # Both are scalars
        inner_product = x_j * theta
    else:
        inner_product = np.dot(x_j, theta) if x_j.ndim > 0 and theta.ndim > 0 else x_j * theta
    # Convert to scalar if it's a NumPy array
    properties["inner_product"] = float(inner_product.item() if isinstance(inner_product, np.ndarray) else inner_product)

    # Induced norm of x_j
    if x_j.ndim == 0:
        induced_norm_xj = np.sqrt(x_j * x_j)
    else:
        induced_norm_xj = np.sqrt(np.dot(x_j, x_j))
    properties["induced_norm_xj"] = float(induced_norm_xj.item() if isinstance(induced_norm_xj, np.ndarray) else induced_norm_xj)

    # Induced metric for xj_minus_theta
    if xj_minus_theta.ndim == 0:
        induced_metric_xj_theta = np.sqrt(xj_minus_theta * xj_minus_theta)
    else:
        induced_metric_xj_theta = np.sqrt(np.dot(xj_minus_theta, xj_minus_theta))
    properties["induced_metric_xj_theta"] = float(induced_metric_xj_theta.item() if isinstance(induced_metric_xj_theta, np.ndarray) else induced_metric_xj_theta)

    # Transformations and Embeddings
    properties["embedding_method"] = selected_embedding
    properties["embedding_dimensions"] = doc_vectors_transformed.shape[1]

    # Scale and Units
    properties["scale_note"] = "Features standardized using StandardScaler (mean=0, std=1)"

    # Underlying Geometry/Topology
    point_cloud = doc_vectors_transformed
    diagram = ripser.ripser(point_cloud)['dgms']
    properties["persistent_homology"] = f"Number of H0 components: {len(diagram[0])}, H1 components: {len(diagram[1]) if len(diagram) > 1 else 0}"
    properties["geometry_note"] = "Data embedded in a vector space with Euclidean geometry, transformed using " + selected_embedding

    return properties

# Example formulas for j and theta
feature_names = [f"x_{i}" for i in range(doc_vectors_transformed.shape[1])]
j_formula = "x_0 + x_1 * (x_2 - x_3) / x_4 if len(feature_names) > 4 else x_0"
theta_formula = "x_0 - x_2 + unknown_var"

# Parse and validate j and theta
print("Validating j and theta formulas...")
identified_operators_j, operator_languages_j, variables_j, unidentified_j = parse_formula_sympy(j_formula, operator_to_languages, feature_names)
identified_operators_theta, operator_languages_theta, variables_theta, unidentified_theta = parse_formula_sympy(theta_formula, operator_to_languages, feature_names)

print("\nIdentified Operators in j:")
for op in identified_operators_j:
    print(f"Operator: {op}, Supported Languages: {operator_languages_j[op]}")
print(f"Variables: {variables_j}")
print(f"Unidentified Symbols: {unidentified_j}")

print("\nIdentified Operators in theta:")
for op in identified_operators_theta:
    print(f"Operator: {op}, Supported Languages: {operator_languages_theta[op]}")
print(f"Variables: {variables_theta}")
print(f"Unidentified Symbols: {unidentified_theta}")

# Map variables to transformed features
var_mapping = {f"x_{i}": doc_vectors_transformed[:, i] for i in range(doc_vectors_transformed.shape[1])}

# Configurations for unidentified symbols
unidentified_configs = [
    {"unknown_var": 1},
    {"unknown_var": doc_vectors_transformed[:, 0] if doc_vectors_transformed.shape[1] > 0 else np.ones(num_docs)},
    {"unknown_var": 2},
]

# Available linkage methods
linkage_methods = ["ward.D", "ward.D2", "single", "average", "mcquitty", "median", "centroid"]

# Prompt user for linkage method
print("\nAvailable linkage methods for dendrogram generation:")
for i, method in enumerate(linkage_methods, 1):
    print(f"{i}. {method}")
while True:
    try:
        choice = int(input("Select a linkage method (enter the number): ")) - 1
        if 0 <= choice < len(linkage_methods):
            selected_linkage = linkage_methods[choice]
            break
        else:
            print(f"Please enter a number between 1 and {len(linkage_methods)}.")
    except ValueError:
        print("Please enter a valid number.")

print(f"\nSelected linkage method: {selected_linkage}")

# Convert j and theta to TeX
j_tex = formula_to_tex(j_formula, operator_to_tex, feature_names)
theta_tex = formula_to_tex(theta_formula, operator_to_tex, feature_names)
print(f"\nTeX representation of j: ${j_tex}$")
print(f"TeX representation of theta: ${theta_tex}$")

best_measures = {}
tex_steps = []
metric_validity = {}  # Store metric validity results

# Loop over different values of j (feature indices)
for j in range(doc_vectors_transformed.shape[1]):
    print(f"\nProcessing feature j={j} (Feature: x_{j})")
    j_vec = doc_vectors_transformed[:, j]

    for config_idx, unid_values in enumerate(unidentified_configs):
        config_name = f"Feature j={j}, Config {config_idx + 1}"
        print(f"\nProcessing Configuration: {config_name}")

        # Compute theta as a distance-based measure
        theta_data = np.zeros(num_docs)
        # Placeholder for best_divergence; will be updated after divergences are computed
        divergences = {}
        best_divergence = selected_metrics[0] if selected_metrics else "Euclidean"  # Default to first metric

        # Prepare vectors for distance computation
        j_vec = j_vec.reshape(-1, 1)
        doc_probs = normalize(doc_vectors_transformed, norm='l1', axis=1)

        # --- Compute Selected Distance Metrics ---
        existing_distances = {}
        cov_matrix = np.cov(doc_vectors_transformed.T) + np.eye(doc_vectors_transformed.shape[1]) * 1e-10

        for metric in selected_metrics:
            if metric == "Euclidean":
                existing_distances[metric] = pairwise_distances(doc_vectors_transformed, metric='euclidean')
            elif metric == "Manhattan":
                existing_distances[metric] = pairwise_distances(doc_vectors_transformed, metric='cityblock')
            elif metric == "Chebyshev":
                existing_distances[metric] = pairwise_distances(doc_vectors_transformed, metric='chebyshev')
            elif metric == "Minkowski":
                existing_distances[metric] = pairwise_distances(doc_vectors_transformed, metric='minkowski', p=3)
            elif metric == "Mahalanobis":
                existing_distances[metric] = pairwise_distances(doc_vectors_transformed, metric='mahalanobis', VI=np.linalg.inv(cov_matrix))
            elif metric == "Cosine":
                existing_distances[metric] = pairwise_distances(doc_vectors_transformed, metric='cosine')
            elif metric == "Hamming":
                doc_vectors_binary = (doc_vectors_transformed > 0).astype(int)
                existing_distances[metric] = pairwise_distances(doc_vectors_binary, metric='hamming')
            elif metric == "Levenshtein":
                dist_matrix = np.zeros((num_docs, num_docs))
                doc_strings = [str(vec) for vec in doc_vectors_transformed]
                for i in range(num_docs):
                    for k in range(num_docs):
                        s1, s2 = doc_strings[i], doc_strings[k]
                        if len(s1) < len(s2):
                            s1, s2 = s2, s1
                        if len(s2) == 0:
                            dist_matrix[i, k] = len(s1)
                            continue
                        previous_row = range(len(s2) + 1)
                        for m, c1 in enumerate(s1):
                            current_row = [m + 1]
                            for n, c2 in enumerate(s2):
                                insertions = previous_row[n + 1] + 1
                                deletions = current_row[n] + 1
                                substitutions = previous_row[n] + (c1 != c2)
                                current_row.append(min(insertions, deletions, substitutions))
                            previous_row = current_row
                        dist_matrix[i, k] = previous_row[-1]
                existing_distances[metric] = dist_matrix
            elif metric == "Alcubierre":
                dist_matrix = np.zeros((num_docs, num_docs))
                for i in range(num_docs):
                    for k in range(num_docs):
                        dist_matrix[i, k] = alcubierre_dist(doc_vectors_transformed[i], doc_vectors_transformed[k], warp_factor=1.0)
                existing_distances[metric] = dist_matrix
            elif metric in ["KL Divergence", "Jensen-Shannon", "Bhattacharyya", "Hellinger", "Wasserstein"]:
                def kl_divergence(p, q):
                    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
                
                def js_divergence(p, q):
                    m = 0.5 * (p + q)
                    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
                
                def bhattacharyya_dist(p, q):
                    return -np.log(np.sum(np.sqrt(p * q)) + 1e-10)
                
                def hellinger_dist(p, q):
                    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
                
                dist_matrix = np.zeros((num_docs, num_docs))
                for i in range(num_docs):
                    for k in range(num_docs):
                        if metric == "KL Divergence":
                            dist_matrix[i, k] = kl_divergence(doc_probs[i], doc_probs[k])
                        elif metric == "Jensen-Shannon":
                            dist_matrix[i, k] = js_divergence(doc_probs[i], doc_probs[k])
                        elif metric == "Bhattacharyya":
                            dist_matrix[i, k] = bhattacharyya_dist(doc_probs[i], doc_probs[k])
                        elif metric == "Hellinger":
                            dist_matrix[i, k] = hellinger_dist(doc_probs[i], doc_probs[k])
                        elif metric == "Wasserstein":
                            cost_matrix = ot.dist(doc_vectors_transformed[[i]], doc_vectors_transformed[[k]])
                            dist_matrix[i, k] = ot.emd2(doc_probs[i], doc_probs[k], cost_matrix)
                existing_distances[metric] = dist_matrix
            elif metric == "Persistent Homology":
                dist_matrix = compute_persistent_homology_distance(doc_vectors_transformed)
                existing_distances[metric] = dist_matrix
            elif metric == "Neural Network":
                dist_matrix = compute_neural_distance(doc_vectors_transformed, neural_model)
                existing_distances[metric] = dist_matrix
            elif metric == "Fibonacci":
                dist_matrix = np.zeros((num_docs, num_docs))
                for i in range(num_docs):
                    for k in range(num_docs):
                        dist_matrix[i, k] = fibonacci_distance(documents[i].split(), documents[k].split())
                existing_distances[metric] = dist_matrix

        # --- Check Metric Validity for Existing Distances ---
        for name, dist_matrix in existing_distances.items():
            valid, reason = is_metric(dist_matrix)
            metric_validity[name] = (valid, reason)
            print(f"{name} → Metric: {valid} ({reason})")

        # --- Phase 2: Entropies ---
        print(f"\n--- Phase 2: Computing Entropies for {config_name} ---")
        fisher_info = np.var(np.log(doc_probs + 1e-10), axis=1)
        q = 2
        tsallis_entropy = (1 - np.sum(doc_probs ** q, axis=1)) / (q - 1)
        shannon_entropy = -np.sum(doc_probs * np.log(doc_probs + 1e-10), axis=1)
        kolmogorov_complexity = np.array([len(str(vec)) for vec in doc_vectors_transformed])
        kolmogorov_entropy = shannon_entropy
        density_matrices = [np.diag(p) for p in doc_probs]
        von_neumann_entropy = np.array([-np.trace(m @ np.log(m + 1e-10)) for m in density_matrices])
        quantum_entropy = von_neumann_entropy

        # --- Discover New Distance Formulas ---
        print(f"\nDiscovering new distance formulas for {config_name}...")
        new_distances = discover_new_distance_formulas(operators_df, operator_to_tex, existing_distances, num_fuzzed=10)
        tex_steps.append(f"\\subsection{{New Distance Formulas for {config_name}}}")
        for name, dist_info in new_distances.items():
            tex_steps.append(f"\\textbf{{{name}}}: ${dist_info['tex']}$")
            dist_matrix = np.zeros((num_docs, num_docs))
            for i in range(num_docs):
                for k in range(num_docs):
                    dist_matrix[i, k] = dist_info["function"](i, k)
            existing_distances[name] = dist_matrix
            # Check metric validity for new distances
            valid, reason = is_metric(dist_matrix)
            metric_validity[name] = (valid, reason)
            print(f"{name} → Metric: {valid} ({reason})")

        # --- Metric Evaluation ---
        entropies = {
            "Shannon": shannon_entropy,
            "Tsallis": tsallis_entropy,
            "Fisher": fisher_info,
            "Von Neumann": von_neumann_entropy,
            "Quantum": quantum_entropy,
            "Kolmogorov Complexity": kolmogorov_complexity,
            "Kolmogorov Entropy": kolmogorov_entropy
        }

        divergences = existing_distances

        entropy_scores = {}
        for name, entropy in entropies.items():
            variance, mutual_info = compute_entropy_metrics(entropy, true_labels)
            composite = compute_composite_score(0, 0, mutual_info, is_entropy=True)
            entropy_scores[name] = composite
        selected_entropies = [name for name, score in entropy_scores.items() if score > np.mean(list(entropy_scores.values()))]

        divergence_scores = {}
        linkage_results = {method: {} for method in linkage_methods}
        for name, divergence in divergences.items():
            for method in linkage_methods:
                # Ward linkage does not support precomputed distances, so we convert to a compatible linkage
                linkage_method = "average" if method in ["ward.D", "ward.D2"] else method
                try:
                    # Compute linkage for dendrogram
                    linked = linkage(divergence, method=linkage_method)
                    # Update AgglomerativeClustering to use 'metric' instead of 'affinity'
                    clustering = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average')
                    cluster_labels = clustering.fit_predict(divergence)
                    if len(np.unique(cluster_labels)) < 2:
                        silhouette, davies_bouldin, mutual_info = -1, float('inf'), 0
                    else:
                        silhouette = silhouette_score(divergence, cluster_labels, metric='precomputed')
                        try:
                            davies_bouldin = davies_bouldin_score(divergence, cluster_labels)
                        except:
                            davies_bouldin = float('inf')  # Handle edge cases
                        mutual_info = mutual_info_score(true_labels, cluster_labels)
                    composite = compute_composite_score(silhouette, davies_bouldin, mutual_info)
                    linkage_results[method][name] = {
                        "silhouette": silhouette,
                        "davies_bouldin": davies_bouldin,
                        "mutual_info": mutual_info,
                        "composite_score": composite if composite != -1 else -np.inf
                    }
                except ValueError as e:
                    print(f"Error with linkage method {method} for {name}: {e}")
                    linkage_results[method][name] = {
                        "silhouette": -1,
                        "davies_bouldin": float('inf'),
                        "mutual_info": 0,
                        "composite_score": -np.inf
                    }
            divergence_scores[name] = linkage_results[selected_linkage][name]["composite_score"]

        selected_divergences = [name for name, score in divergence_scores.items() if score > np.mean([s for s in divergence_scores.values() if s != -np.inf])]

        # Update best_divergence now that we have divergence scores
        best_divergence = max(divergence_scores, key=divergence_scores.get)

        # Compute theta using the best divergence metric
        for doc_idx in range(num_docs):
            theta_data[doc_idx] = compute_theta_as_distance(doc_vectors_transformed, best_divergence, divergences, doc_idx)
        theta_data = theta_data / (theta_data.max() + 1e-10)  # Normalize

        # Compute properties for the main formula
        formula_properties = []
        for doc_idx in range(num_docs):
            x_j_val = j_vec[doc_idx]
            theta_val = theta_data[doc_idx]
            props = compute_formula_properties(x_j_val, theta_val, doc_vectors_transformed, selected_embedding, j)
            formula_properties.append(props)

        # --- Univariate Assessment ---
        j_stats = univariate_assessment(j_vec.flatten(), "j")
        theta_stats = univariate_assessment(theta_data, "theta")
        print(f"\nUnivariate Assessment for j ({config_name}):")
        print(f"  Mean: {j_stats['mean']:.4f}")
        print(f"  Variance: {j_stats['variance']:.4f}")
        print(f"  Skewness: {j_stats['skewness']:.4f}")
        print(f"  Kurtosis: {j_stats['kurtosis']:.4f}")
        print(f"  Shannon Entropy: {j_stats['shannon_entropy']:.4f}")
        print(f"\nUnivariate Assessment for theta ({config_name}):")
        print(f"  Mean: {theta_stats['mean']:.4f}")
        print(f"  Variance: {theta_stats['variance']:.4f}")
        print(f"  Skewness: {theta_stats['skewness']:.4f}")
        print(f"  Kurtosis: {theta_stats['kurtosis']:.4f}")
        print(f"  Shannon Entropy: {theta_stats['shannon_entropy']:.4f}")

        # --- Multivariate Assessment ---
        multi_stats = multivariate_assessment(j_vec.flatten(), theta_data)
        print(f"\nMultivariate Assessment for j and theta ({config_name}):")
        print(f"  Covariance: {multi_stats['covariance']:.4f}")
        print(f"  Correlation: {multi_stats['correlation']:.4f}")
        print(f"  Mutual Information: {multi_stats['mutual_info']:.4f}")

        # --- Commutation of j and theta ---
        j_theta_commuted = commute_j_theta(j_vec.flatten(), theta_data)
        print(f"\nCommuted j o theta ({config_name}): {j_theta_commuted}")
        threshold_commuted = apply_threshold(j_vec.flatten(), theta_data, R)
        print(f"Threshold with commuted j o theta: {threshold_commuted}")

        # Apply threshold
        selected_entropy_docs = {}
        for name, entropy in entropies.items():
            theta_entropy = entropy / (entropy.max() + 1e-10)
            threshold = apply_threshold(j_vec.flatten(), theta_entropy, R)
            selected_entropy_docs[name] = np.where(threshold)[0]

        selected_divergence_pairs = {}
        for name, divergence in divergences.items():
            theta_divergence = divergence / (divergence.max() + 1e-10)
            threshold = apply_threshold(j_vec, theta_divergence, R)
            selected_divergence_pairs[name] = np.where(threshold)

        # Render final equation in TeX
        tex_steps.append(f"\\subsection{{Final Equation for {config_name}}}")
        final_equation = f"x_j > R * (1 - theta)"
        final_equation_tex = f"x_{{{j}}} \\geq \\mathbb{{R}} \\times (1 - \\theta)"
        # Update theta to include the distance function
        theta_tex = f"\\frac{{1}}{{N}} \\sum_{{k=1}}^{{N}} d_{{ \\text{{{best_divergence}}}}} (x_{{i}}, x_{{k}})"
        final_equation_tex = final_equation_tex.replace("\\theta", theta_tex)
        tex_steps.append(f"Final threshold equation: ${final_equation_tex}$")
        tex_steps.append(f"Evaluated threshold: ${{{final_equation_tex}}}$, Result: ${threshold_commuted}$")

        # Add formula properties to TeX
        tex_steps.append(f"\\subsection{{Properties of the Main Formula for {config_name}}}")
        for doc_idx in range(num_docs):
            props = formula_properties[doc_idx]
            tex_steps.append(f"\\subsubsection{{Document {doc_labels[doc_idx]}}}")
            tex_steps.append("\\begin{itemize}")
            # Norms
            tex_steps.append(f"\\item \\textbf{{Norms for $x_{{{j}}}$:}}")
            tex_steps.append(f"  \\begin{{itemize}}")
            tex_steps.append(f"    \\item $L_1$ Norm: ${props['L1_norm_xj']:.4f}$")
            tex_steps.append(f"    \\item $L_2$ Norm: ${props['L2_norm_xj']:.4f}$")
            tex_steps.append(f"    \\item $L_\\infty$ Norm: ${props['Linf_norm_xj']:.4f}$")
            tex_steps.append(f"  \\end{{itemize}}")
            tex_steps.append(f"\\item \\textbf{{Norms for $x_{{{j}}} - \\theta$:}}")
            tex_steps.append(f"  \\begin{{itemize}}")
            tex_steps.append(f"    \\item $L_1$ Norm: ${props['L1_norm_xj_minus_theta']:.4f}$")
            tex_steps.append(f"    \\item $L_2$ Norm: ${props['L2_norm_xj_minus_theta']:.4f}$")
            tex_steps.append(f"    \\item $L_\\infty$ Norm: ${props['Linf_norm_xj_minus_theta']:.4f}$")
            tex_steps.append(f"  \\end{{itemize}}")
            # Inner Products
            tex_steps.append(f"\\item \\textbf{{Inner Product:}} $\\langle x_{{{j}}}, \\theta \\rangle = {props['inner_product']:.4f}$")
            tex_steps.append(f"\\item \\textbf{{Induced Norm of $x_{{{j}}}$:}} $\\sqrt{{\\langle x_{{{j}}}, x_{{{j}}} \\rangle}} = {props['induced_norm_xj']:.4f}$")
            tex_steps.append(f"\\item \\textbf{{Induced Metric:}} $\\sqrt{{\\langle x_{{{j}}} - \\theta, x_{{{j}}} - \\theta \\rangle}} = {props['induced_metric_xj_theta']:.4f}$")
            # Transformations and Embeddings
            tex_steps.append(f"\\item \\textbf{{Transformations and Embeddings:}} {props['embedding_method']} with {props['embedding_dimensions']} dimensions")
            # Scale and Units
            tex_steps.append(f"\\item \\textbf{{Scale and Units:}} {props['scale_note']}")
            # Geometry/Topology
            tex_steps.append(f"\\item \\textbf{{Underlying Geometry/Topology:}}")
            tex_steps.append(f"  \\begin{{itemize}}")
            tex_steps.append(f"    \\item Persistent Homology: {props['persistent_homology']}")
            tex_steps.append(f"    \\item Geometry Note: {props['geometry_note']}")
            tex_steps.append(f"  \\end{{itemize}}")
            tex_steps.append("\\end{itemize}")

        # Store results
        config_results = {}
        for name, divergence in divergences.items():
            metrics = linkage_results[selected_linkage][name]
            config_results[name] = {
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "mutual_info": metrics["mutual_info"],
                "composite_score": metrics["composite_score"]
            }

        variance, entropy_mi = compute_entropy_metrics(shannon_entropy, true_labels)
        entropy_score = compute_composite_score(0, 0, entropy_mi, is_entropy=True)

        best_measures[config_name] = {
            "distance_scores": config_results,
            "entropy_score": entropy_score,
            "variance": variance,
            "entropy_mi": entropy_mi,
            "best_entropy": max(entropy_scores, key=entropy_scores.get),
            "best_entropy_score": max(entropy_scores.values()),
            "best_divergence": max(divergence_scores, key=divergence_scores.get),
            "best_divergence_score": max(divergence_scores.values()),
            "linkage_results": linkage_results,
            "j_stats": j_stats,
            "theta_stats": theta_stats,
            "multi_stats": multi_stats,
            "j_theta_commuted": j_theta_commuted,
            "threshold_commuted": threshold_commuted,
            "formula_properties": formula_properties
        }

        # --- Visualization ---
        for name, entropy in entropies.items():
            selected_docs = selected_entropy_docs[name] if name in selected_entropies else None
            plot_entropy(entropy, f"{config_name}: {name} Entropy (Selected: {name in selected_entropies})", doc_labels, selected_docs)

        for name, divergence in divergences.items():
            title = f"{config_name}: {name} Heatmap (Selected: {name in selected_divergences})"
            plot_heatmap(divergence, title, doc_labels)
            if name in selected_divergences:
                plot_dendrogram(divergence, f"{config_name}: {name} Dendrogram", doc_labels, selected_linkage)

        selected_div = best_measures[config_name]["best_divergence"]
        theta_selected = divergences[selected_div] / (divergences[selected_div].max() + 1e-10)
        threshold_selected = apply_threshold(j_vec, theta_selected, R)

        plt.figure(figsize=(8, 6))
        for i in range(num_docs):
            plt.bar(doc_labels[i], j_vec[i], color='green' if threshold_selected[i, i] else 'red')
        plt.title(f"{config_name}: Feature Values (Green: Satisfies Formula with {selected_div}, Red: Does Not)")
        plt.xlabel("Documents")
        plt.ylabel("Feature Value")
        plt.xticks(rotation=45)
        plt.show()

        # --- Visualize Metrics Usage ---
        plot_metrics_usage(selected_metrics, all_metrics, config_name)
        tex_steps.append(f"\\subsection{{Metrics Usage for {config_name}}}")
        tex_steps.append(f"\\includegraphics[width=\\textwidth]{{metrics_usage_{config_name.replace(' ', '_')}.png}}")

# --- Metric Validity Summary ---
print("\n=== Metric Validity Summary ===")
for name, (valid, reason) in metric_validity.items():
    print(f"{name}: {'✅' if valid else '❌'} ({reason})")

# Add metric validity to TeX report
tex_steps.append("\\section{Metric Validity Summary}")
tex_steps.append("\\begin{itemize}")
for name, (valid, reason) in metric_validity.items():
    status = "\\checkmark" if valid else "\\times"
    tex_steps.append(f"\\item \\textbf{{{name}}}: ${status}$ ({reason})")
tex_steps.append("\\end{itemize}")

# --- Summary ---
print("\n=== Summary of Best Measures ===")
for config, results in best_measures.items():
    print(f"\n{config}:")
    for name, metrics in results["distance_scores"].items():
        print(f"  {name}: Composite Score = {metrics['composite_score']:.3f} "
              f"(Silhouette: {metrics['silhouette']:.3f}, DB: {metrics['davies_bouldin']:.3f}, MI: {metrics['mutual_info']:.3f})")
    print(f"  Entropy Score = {results['entropy_score']:.3f} "
          f"(Variance = {results['variance']:.4f}, MI = {results['entropy_mi']:.3f})")
    print(f"  Best Entropy: {results['best_entropy']} (Score: {results['best_entropy_score']:.4f})")
    print(f"  Best Divergence: {results['best_divergence']} (Score: {results['best_divergence_score']:.4f})")

# --- Linkage Method Comparison ---
print("\n=== Linkage Method Comparison ===")
for method in linkage_methods:
    print(f"\nLinkage Method: {method}")
    for config, results in best_measures.items():
        print(f"  {config}:")
        for name, metrics in results["linkage_results"][method].items():
            print(f"    {name}: Composite Score = {metrics['composite_score']:.3f}")

# --- Composite Scores Visualization ---
plt.figure(figsize=(10, 6))
configs = list(best_measures.keys())
entropy_scores = [best_measures[config]["entropy_score"] for config in configs]
divergence_scores = [best_measures[config]["best_divergence_score"] for config in configs]

x = np.arange(len(configs))
width = 0.35
plt.bar(x - width/2, entropy_scores, width, label='Entropy Score', color='skyblue')
plt.bar(x + width/2, divergence_scores, width, label='Best Divergence Score', color='salmon')
plt.xlabel("Configurations")
plt.ylabel("Composite Scores")
plt.title("Entropy and Best Divergence Scores by Configuration")
plt.xticks(x, configs, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# --- Export TeX Steps to File ---
with open("distance_formulas.tex", "w") as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{amsmath}\n")
    f.write("\\usepackage{amssymb}\n")
    f.write("\\usepackage{graphicx}\n")
    f.write("\\begin{document}\n")
    f.write("\\title{New Distance Formulas}\n")
    f.write("\\maketitle\n")
    f.write("\\section{Document Analysis}\n")
    f.write(f"Total documents processed: {num_docs}\n")
    f.write("\\begin{itemize}\n")
    for doc in doc_labels:
        f.write(f"\\item {doc}\n")
    f.write("\\end{itemize}\n")
    f.write(f"Total unique terms: {num_terms}\n")
    f.write("\\begin{itemize}\n")
    for i, term in enumerate(vocabulary[:10]):
        f.write(f"\\item Term {i}: {term}\n")
    if num_terms > 10:
        f.write(f"\\item ... (and {num_terms - 10} more terms)\n")
    f.write("\\end{itemize}\n")
    f.write("\\section{Available Metrics}\n")
    f.write("\\begin{itemize}\n")
    for metric, info in all_metrics.items():
        status = "Implemented" if info["implemented"] else "Not Implemented"
        f.write(f"\\item \\textbf{{{metric}}} ({info['space']}, {status}): {info['description']}\n")
    f.write("\\end{itemize}\n")
    f.write("\\section{Selected Metrics}\n")
    f.write("\\begin{itemize}\n")
    for metric in selected_metrics:
        f.write(f"\\item {metric}\n")
    f.write("\\end{itemize}\n")
    f.write("\\section{Analysis Steps}\n")
    for step in tex_steps:
        f.write(step + "\n")
    f.write("\\end{document}\n")
print("\nTeX steps exported to 'distance_formulas.tex'")

# --- Export Results to CSV ---
results_df = []
for config, results in best_measures.items():
    for name, metrics in results["distance_scores"].items():
        is_valid, reason = metric_validity.get(name, (False, "Not evaluated"))
        results_df.append({
            "Configuration": config,
            "Metric": name,
            "Type": "Divergence",
            "Composite Score": metrics["composite_score"],
            "Silhouette": metrics["silhouette"],
            "Davies-Bouldin": metrics["davies_bouldin"],
            "Mutual Info": metrics["mutual_info"],
            "Is Metric": is_valid,
            "Metric Reason": reason
        })
    results_df.append({
        "Configuration": config,
        "Metric": "Entropy",
        "Type": "Entropy",
        "Composite Score": results["entropy_score"],
        "Variance": results["variance"],
        "Mutual Info": results["entropy_mi"],
        "Silhouette": None,
        "Davies-Bouldin": None,
        "Is Metric": None,
        "Metric Reason": None
    })
    results_df.append({
        "Configuration": config,
        "Metric": "j_mean",
        "Type": "Univariate",
        "Value": results["j_stats"]["mean"],
        "Is Metric": None,
        "Metric Reason": None
    })
    results_df.append({
        "Configuration": config,
        "Metric": "j_variance",
        "Type": "Univariate",
        "Value": results["j_stats"]["variance"],
        "Is Metric": None,
        "Metric Reason": None
    })
    results_df.append({
        "Configuration": config,
        "Metric": "theta_mean",
        "Type": "Univariate",
        "Value": results["theta_stats"]["mean"],
        "Is Metric": None,
        "Metric Reason": None
    })
    results_df.append({
        "Configuration": config,
        "Metric": "j_theta_correlation",
        "Type": "Multivariate",
        "Value": results["multi_stats"]["correlation"],
        "Is Metric": None,
        "Metric Reason": None
    })

results_df = pd.DataFrame(results_df)
results_df.to_csv("evaluation_results.csv", index=False)
print("\nResults exported to 'evaluation_results.csv'")

# --- Ranking Across Configurations ---
print("\n=== Ranking of Configurations ===")
divergence_ranking = sorted(
    [(config, results["best_divergence"], results["best_divergence_score"]) 
     for config, results in best_measures.items()],
    key=lambda x: x[2],
    reverse=True
)
print("\nDivergence Ranking:")
for rank, (config, best_div, score) in enumerate(divergence_ranking, 1):
    is_valid, reason = metric_validity.get(best_div, (False, "Not evaluated"))
    print(f"Rank {rank}: {config} (Best Divergence: {best_div}, Score: {score:.4f}, Metric: {'✅' if is_valid else '❌'} ({reason}))")

entropy_ranking = sorted(
    [(config, results["entropy_score"]) 
     for config, results in best_measures.items()],
    key=lambda x: x[1],
    reverse=True
)
print("\nEntropy Ranking:")
for rank, (config, score) in enumerate(entropy_ranking, 1):
    print(f"Rank {rank}: {config} (Entropy Score: {score:.4f})")
