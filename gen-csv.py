#!/usr/bin/env python3
# By Thibaut LOMBARD (@lombardweb)
# gen-csv.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cosine, cityblock, hamming, minkowski
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import skew, kurtosis
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score, mutual_info_score
from sklearn.cluster import AgglomerativeClustering
import ot
import sympy as sp
import re
import warnings
warnings.filterwarnings("ignore")

# --- Step 1: Load documents from the "documents" folder ---
documents_folder = "documents"
if not os.path.exists(documents_folder):
    os.makedirs(documents_folder)
    print(f"Created folder '{documents_folder}'. Please add .txt files to this folder and rerun the script.")
    exit()

# Get all .txt files in the documents folder
txt_files = [f for f in os.listdir(documents_folder) if f.endswith('.txt')]
num_docs = len(txt_files)

if num_docs == 0:
    print(f"No .txt files found in the '{documents_folder}' folder. Please add .txt files and rerun the script.")
    exit()

print(f"Found {num_docs} documents in the '{documents_folder}' folder.")

# Read the content of each document and build a vocabulary
documents = []
doc_labels = []
all_terms = set()

for i, file_name in enumerate(txt_files):
    with open(os.path.join(documents_folder, file_name), 'r', encoding='utf-8') as f:
        content = f.read().lower()  # Convert to lowercase for consistency
        documents.append(content)
        doc_labels.append(file_name.replace('.txt', ''))
        # Split content into terms (words) and add to vocabulary
        terms = re.findall(r'\b\w+\b', content)
        all_terms.update(terms)

# Create a list of unique terms (vocabulary)
vocabulary = sorted(list(all_terms))
num_terms = len(vocabulary)
print(f"Total unique terms in vocabulary: {num_terms}")

# Compute document frequency for each term
doc_freqs = np.zeros(num_terms)  # Document frequency for each term
term_to_index = {term: idx for idx, term in enumerate(vocabulary)}

for doc in documents:
    terms_in_doc = set(re.findall(r'\b\w+\b', doc.lower()))
    for term in terms_in_doc:
        if term in term_to_index:
            doc_freqs[term_to_index[term]] += 1

# Simulate document labels for mutual information (since we don't have true labels)
np.random.seed(42)
true_labels = np.random.randint(0, 3, size=num_docs)  # Random labels for clustering evaluation

N = num_docs

# Load and parse the operators table
operators_df = pd.read_csv("operators_table.csv")
languages = operators_df.columns[1:-2]  # Last column is 'TeX', second-to-last is 'Description'

# Map operators to languages and their TeX representations
operator_to_languages = {}
operator_to_tex = {}
for _, row in operators_df.iterrows():
    operator = row['Operator']
    operator_to_languages[operator] = []
    for lang in languages:
        value = str(row[lang]).strip()
        if value.startswith("Yes"):
            operator_to_languages[operator].append(lang)
    operator_to_tex[operator] = row['TeX']  # Last column is TeX

# Parse a formula and validate it
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

# Validate and evaluate a formula using SymPy
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

# Generate TeX representation of a formula
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

# Discover new distance formulas by combining operators
def discover_new_distance_formulas(operators_df, operator_to_tex, existing_distances):
    new_distances = {}
    operator_list = operators_df['Operator'].tolist()
    
    distance_names = list(existing_distances.keys())
    for i, dist1_name in enumerate(distance_names):
        for dist2_name in distance_names[i+1:]:
            for op in operator_list:
                if op in ['&&', '||', '>', '<', '>=', '<=', '==', '!=']:
                    continue
                new_dist_name = f"{dist1_name}_{op}_{dist2_name}"
                try:
                    tex_op = operator_to_tex.get(op, op)
                    tex_desc = f"d_{{ \\text{{{new_dist_name}}}} (u, v) = d_{{ \\text{{{dist1_name}}}} (u, v) {tex_op} d_{{ \\text{{{dist2_name}}}} (u, v)"
                    
                    def new_dist_func(u, v, dist1=existing_distances[dist1_name], dist2=existing_distances[dist2_name], operator=op):
                        d1 = dist1[u, v]
                        d2 = dist2[u, v]
                        if operator == '+':
                            return d1 + d2
                        elif operator == '-':
                            return np.abs(d1 - d2)
                        elif operator == '*':
                            return d1 * d2
                        elif operator == '/':
                            return d1 / (d2 + 1e-10)
                        elif operator == '^':
                            return d1 ** d2 if d2 >= 0 else 0
                        else:
                            return d1 + d2
                    
                    new_distances[new_dist_name] = {
                        "function": new_dist_func,
                        "tex": tex_desc
                    }
                except Exception as e:
                    print(f"Error creating new distance {new_dist_name}: {e}")
    
    return new_distances

# Threshold formula: d_fj > N * (1 - theta)
def apply_threshold(d_fj, theta, N):
    return d_fj > N * (1 - theta)

# Univariate assessment
def univariate_assessment(data, name):
    mean = np.mean(data)
    variance = np.var(data)
    skewness = skew(data)
    kurt = kurtosis(data)
    return {
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurt,
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

# Compute evaluation metrics
def compute_clustering_metrics(dist_matrix, true_labels):
    if np.all(dist_matrix == 0):
        return -1, float('inf'), 0
    clustering = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average')
    cluster_labels = clustering.fit_predict(dist_matrix)
    if len(np.unique(cluster_labels)) < 2:
        return -1, float('inf'), 0
    
    silhouette = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
    davies_bouldin = davies_bouldin_score(dist_matrix, cluster_labels)
    mutual_info = mutual_info_score(true_labels, cluster_labels)
    return silhouette, davies_bouldin, mutual_info

def compute_entropy_metrics(entropy, true_labels):
    variance = np.var(entropy)
    entropy_discrete = np.digitize(entropy, bins=np.linspace(entropy.min(), entropy.max(), 5))
    mutual_info = mutual_info_score(true_labels, entropy_discrete)
    return variance, mutual_info

# Composite score
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

# Example formulas for j and theta
# j is now the index of a term, so we redefine the formula context
# Let's assume x1, x2, ..., x5 are document frequencies of the first 5 terms
feature_names = [f"df_{i}" for i in range(min(5, num_terms))]  # Document frequencies as features
j_formula = "df_0 + df_1 * (df_2 - df_3) / df_4 and (df_0 > df_1) or unknown_var ** 2 + @unidentified"
theta_formula = "df_0 - df_2 + unknown_var"

# Parse and validate j and theta
print("Validating j and theta formulas...")
identified_operators_j, operator_languages_j, variables_j, unidentified_j = parse_formula_sympy(j_formula, operator_to_languages, feature_names)
identified_operators_theta, operator_languages_theta, variables_theta, unidentified_theta = parse_formula_sympy(theta_formula, operator_to_languages, feature_names)

# Print operators and supported languages
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

# Map variables to document frequencies
var_mapping = {f"df_{i}": doc_freqs[i] for i in range(min(5, num_terms))}

# Configurations for unidentified symbols
unidentified_configs = [
    {"unknown_var": 1, "@unidentified": 0},
    {"unknown_var": doc_freqs[0] if num_terms > 0 else np.ones(num_docs), "@unidentified": 0},
    {"unknown_var": 2, "@unidentified": 1},
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

# Loop over different values of j (term indices)
for j in range(min(5, num_terms)):  # Limit to first 5 terms or total terms if less
    term_j = vocabulary[j]
    d_fj = doc_freqs[j]  # Document frequency of term at index j
    print(f"\nProcessing term j={j} (Term: {term_j}), Document Frequency (d_fj): {d_fj}")

    for config_idx, unid_values in enumerate(unidentified_configs):
        config_name = f"Term j={j}, Config {config_idx + 1}"
        print(f"\nProcessing Configuration: {config_name}")

        # Evaluate theta
        theta_data = validate_and_evaluate_formula(theta_formula, doc_freqs[:min(5, num_terms)], var_mapping, unid_values, feature_names, operator_to_languages, name="theta")
        if theta_data is None:
            print(f"Skipping configuration {config_name} due to invalid theta formula.")
            continue
        theta_data = theta_data / (theta_data.max() + 1e-10)  # Normalize theta

        # Since d_fj is a scalar (document frequency of term j), we need to create a vector for distance computations
        # We'll replicate d_fj across documents for compatibility with distance metrics
        d_fj_vec = np.full(num_docs, d_fj)

        # Normalize for probability-based measures
        doc_probs = np.ones((num_docs, 1)) / num_docs  # Uniform distribution for simplicity

        # --- Univariate Assessment ---
        j_stats = univariate_assessment(d_fj_vec, "j")
        theta_stats = univariate_assessment(theta_data, "theta")
        print(f"\nUnivariate Assessment for j ({config_name}):")
        print(f"  Mean: {j_stats['mean']:.4f}")
        print(f"  Variance: {j_stats['variance']:.4f}")
        print(f"  Skewness: {j_stats['skewness']:.4f}")
        print(f"  Kurtosis: {j_stats['kurtosis']:.4f}")
        print(f"\nUnivariate Assessment for theta ({config_name}):")
        print(f"  Mean: {theta_stats['mean']:.4f}")
        print(f"  Variance: {theta_stats['variance']:.4f}")
        print(f"  Skewness: {theta_stats['skewness']:.4f}")
        print(f"  Kurtosis: {theta_stats['kurtosis']:.4f}")

        # --- Multivariate Assessment ---
        multi_stats = multivariate_assessment(d_fj_vec, theta_data)
        print(f"\nMultivariate Assessment for j and theta ({config_name}):")
        print(f"  Covariance: {multi_stats['covariance']:.4f}")
        print(f"  Correlation: {multi_stats['correlation']:.4f}")
        print(f"  Mutual Information: {multi_stats['mutual_info']:.4f}")

        # --- Commutation of j and theta ---
        j_theta_commuted = commute_j_theta(d_fj_vec, theta_data)
        print(f"\nCommuted j o theta ({config_name}): {j_theta_commuted}")
        threshold_commuted = apply_threshold(j_theta_commuted, theta_data, N)
        print(f"Threshold with commuted j o theta: {threshold_commuted}")

        # --- Phase 1: Classical Distances ---
        d_fj_vec = d_fj_vec.reshape(-1, 1)
        euclidean_dist = pairwise_distances(d_fj_vec, metric='euclidean')
        cosine_dist = pairwise_distances(d_fj_vec, metric='cosine')
        doc_freqs_binary = (d_fj_vec > 0).astype(int)
        hamming_dist = pairwise_distances(doc_freqs_binary, metric='hamming')
        manhattan_dist = pairwise_distances(d_fj_vec, metric='cityblock')
        minkowski_dist = pairwise_distances(d_fj_vec, metric='minkowski', p=3)

        levenshtein_dist_matrix = np.zeros((num_docs, num_docs))
        doc_freqs_str = [str(int(val)) for val in d_fj_vec.flatten()]
        for i in range(num_docs):
            for k in range(num_docs):
                s1, s2 = doc_freqs_str[i], doc_freqs_str[k]
                if len(s1) < len(s2):
                    s1, s2 = s2, s1
                if len(s2) == 0:
                    levenshtein_dist_matrix[i, k] = len(s1)
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
                levenshtein_dist_matrix[i, k] = previous_row[-1]

        alcubierre_dist_matrix = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                alcubierre_dist_matrix[i, k] = alcubierre_dist(d_fj_vec[i], d_fj_vec[k], warp_factor=1.0)

        # --- Phase 2: Divergences & Statistical Entropy ---
        print(f"\n--- Phase 2: Computing Entropies and Divergences for {config_name} ---")

        def kl_divergence(p, q):
            return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

        kl_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                kl_dist[i, k] = kl_divergence(doc_probs[i], doc_probs[k])

        def js_divergence(p, q):
            m = 0.5 * (p + q)
            return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

        js_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                js_dist[i, k] = js_divergence(doc_probs[i], doc_probs[k])

        def bhattacharyya_dist(p, q):
            return -np.log(np.sum(np.sqrt(p * q)) + 1e-10)

        bhattacharyya_dist_matrix = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                bhattacharyya_dist_matrix[i, k] = bhattacharyya_dist(doc_probs[i], doc_probs[k])

        def hellinger_dist(p, q):
            return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

        hellinger_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                hellinger_dist[i, k] = hellinger_dist(doc_probs[i], doc_probs[k])

        def chernoff_divergence(p, q, alpha=0.5):
            return -np.log(np.sum((p ** alpha) * (q ** (1 - alpha))) + 1e-10)

        chernoff_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                chernoff_dist[i, k] = chernoff_divergence(doc_probs[i], doc_probs[k])

        def jeffrey_divergence(p, q):
            return kl_divergence(p, q) + kl_divergence(q, p)

        jeffrey_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                jeffrey_dist[i, k] = jeffrey_divergence(doc_probs[i], doc_probs[k])

        def aitchison_dist(p, q):
            p_clr = np.log(p + 1e-10) - np.mean(np.log(p + 1e-10))
            q_clr = np.log(q + 1e-10) - np.mean(np.log(q + 1e-10))
            return np.sqrt(np.sum((p_clr - q_clr) ** 2))

        aitchison_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                aitchison_dist[i, k] = aitchison_dist(doc_probs[i], doc_probs[k])

        csiszar_dist = kl_dist
        amari_dist = kl_dist

        def bregman_divergence(p, q):
            return np.sum((p - q) ** 2)

        bregman_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                bregman_dist[i, k] = bregman_divergence(doc_probs[i], doc_probs[k])

        cov_matrices = [np.cov(d_fj_vec.T) for _ in range(num_docs)]
        logdet_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                A, B = cov_matrices[i], cov_matrices[k]
                logdet_dist[i, k] = np.trace(A @ np.linalg.pinv(B)) - np.log(np.linalg.det(A @ np.linalg.pinv(B)) + 1e-10) - 1

        def matsushita_dist(p, q):
            return np.sqrt(1 - np.sum(np.sqrt(p * q)))

        matsushita_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                matsushita_dist[i, k] = matsushita_dist(doc_probs[i], doc_probs[k])

        burbea_rao_dist = js_dist

        fisher_info = np.var(np.log(doc_probs + 1e-10), axis=1)
        q = 2
        tsallis_entropy = (1 - np.sum(doc_probs ** q, axis=1)) / (q - 1)
        shannon_entropy = -np.sum(doc_probs * np.log(doc_probs + 1e-10), axis=1)
        kolmogorov_complexity = np.array([len(str(int(val))) for val in d_fj_vec.flatten()])
        kolmogorov_entropy = shannon_entropy

        # --- Phase 3: Quantum & Geometry-Based ---
        density_matrices = [np.diag(p) for p in doc_probs]
        von_neumann_entropy = np.array([-np.trace(m @ np.log(m + 1e-10)) for m in density_matrices])
        frobenius_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                frobenius_dist[i, k] = np.linalg.norm(density_matrices[i] - density_matrices[k], 'fro')
        quantum_entropy = von_neumann_entropy
        hilbert_dist = euclidean_dist
        riemannian_dist = euclidean_dist
        fisher_rao_dist = hellinger_dist

        wasserstein_dist = np.zeros((num_docs, num_docs))
        sinkhorn_dist = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for k in range(num_docs):
                cost_matrix = ot.dist(d_fj_vec[[i]], d_fj_vec[[k]])
                wasserstein_dist[i, k] = ot.emd2(doc_probs[i], doc_probs[k], cost_matrix)
                sinkhorn_dist[i, k] = ot.sinkhorn2(doc_probs[i], doc_probs[k], cost_matrix, reg=1e-1)

        gromov_hausdorff_dist = wasserstein_dist
        ipm_dist = wasserstein_dist

        # --- Discover New Distance Formulas ---
        existing_distances = {
            "Euclidean": euclidean_dist,
            "Cosine": cosine_dist,
            "Hamming": hamming_dist,
            "Manhattan": manhattan_dist,
            "Minkowski": minkowski_dist,
            "Levenshtein": levenshtein_dist_matrix,
            "Alcubierre": alcubierre_dist_matrix,
            "KL Divergence": kl_dist,
            "Jensen-Shannon": js_dist,
            "Bhattacharyya": bhattacharyya_dist_matrix,
            "Hellinger": hellinger_dist,
            "Chernoff": chernoff_dist,
            "Jeffrey": jeffrey_dist,
            "Aitchison": aitchison_dist,
            "Csiszár": csiszar_dist,
            "Amari": amari_dist,
            "Bregman": bregman_dist,
            "LogDet": logdet_dist,
            "Matsushita": matsushita_dist,
            "Burbea-Rao": burbea_rao_dist,
            "Frobenius": frobenius_dist,
            "Hilbert": hilbert_dist,
            "Riemannian": riemannian_dist,
            "Fisher-Rao": fisher_rao_dist,
            "Gromov-Hausdorff": gromov_hausdorff_dist,
            "Sinkhorn": sinkhorn_dist,
            "IPM": ipm_dist,
            "Wasserstein": wasserstein_dist
        }

        print(f"\nDiscovering new distance formulas for {config_name}...")
        new_distances = discover_new_distance_formulas(operators_df, operator_to_tex, existing_distances)
        tex_steps.append(f"\\subsection{{New Distance Formulas for {config_name}}}")
        for name, dist_info in new_distances.items():
            tex_steps.append(f"\\textbf{{{name}}}: ${dist_info['tex']}$")
            dist_matrix = np.zeros((num_docs, num_docs))
            for i in range(num_docs):
                for k in range(num_docs):
                    dist_matrix[i, k] = dist_info["function"](i, k)
            existing_distances[name] = dist_matrix

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
                linkage_method = "ward" if method in ["ward.D", "ward.D2"] else method
                try:
                    linked = linkage(divergence, method=linkage_method)
                    clustering = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average')
                    cluster_labels = clustering.fit_predict(divergence)
                    if len(np.unique(cluster_labels)) < 2:
                        silhouette, davies_bouldin, mutual_info = -1, float('inf'), 0
                    else:
                        silhouette = silhouette_score(divergence, cluster_labels, metric='precomputed')
                        davies_bouldin = davies_bouldin_score(divergence, cluster_labels)
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

        # Apply threshold
        selected_entropy_docs = {}
        for name, entropy in entropies.items():
            theta_entropy = entropy / (entropy.max() + 1e-10)
            threshold = apply_threshold(d_fj_vec.flatten(), theta_entropy, N)
            selected_entropy_docs[name] = np.where(threshold)[0]

        selected_divergence_pairs = {}
        for name, divergence in divergences.items():
            theta_divergence = divergence / (divergence.max() + 1e-10)
            threshold = apply_threshold(d_fj_vec, theta_divergence, N)
            selected_divergence_pairs[name] = np.where(threshold)

        # Render final equation in TeX
        tex_steps.append(f"\\subsection{{Final Equation for {config_name}}}")
        final_equation = f"d_{{f_{j}}} > N \\cdot (1 - \\theta)"
        final_equation_tex = f"d_{{f_{j}}} \\geq {N} \\times (1 - \\theta)"
        final_equation_tex = final_equation_tex.replace(f"d_{{f_{j}}}", f"\\text{{df}}_{{{j}}}")
        final_equation_tex = final_equation_tex.replace("\\theta", theta_tex)
        tex_steps.append(f"Final threshold equation: ${final_equation_tex}$")
        tex_steps.append(f"Evaluated threshold: ${{{final_equation_tex}}}$, Result: ${threshold_commuted}$")

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
            "threshold_commuted": threshold_commuted
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
        threshold_selected = apply_threshold(d_fj_vec, theta_selected, N)

        plt.figure(figsize=(8, 6))
        for i in range(num_docs):
            plt.bar(doc_labels[i], d_fj_vec[i], color='green' if threshold_selected[i, i] else 'red')
        plt.title(f"{config_name}: Document Frequencies (Green: Satisfies Formula with {selected_div}, Red: Does Not)")
        plt.xlabel("Documents")
        plt.ylabel("Formula Output")
        plt.xticks(rotation=45)
        plt.show()

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
    for i, term in enumerate(vocabulary[:10]):  # List first 10 terms
        f.write(f"\\item Term {i}: {term}, Document Frequency: {doc_freqs[i]}\n")
    if num_terms > 10:
        f.write(f"\\item ... (and {num_terms - 10} more terms)\n")
    f.write("\\end{itemize}\n")
    for step in tex_steps:
        f.write(step + "\n")
    f.write("\\end{document}\n")
print("\nTeX steps exported to 'distance_formulas.tex'")

# --- Export Results to CSV ---
results_df = []
for config, results in best_measures.items():
    for name, metrics in results["distance_scores"].items():
        results_df.append({
            "Configuration": config,
            "Metric": name,
            "Type": "Divergence",
            "Composite Score": metrics["composite_score"],
            "Silhouette": metrics["silhouette"],
            "Davies-Bouldin": metrics["davies_bouldin"],
            "Mutual Info": metrics["mutual_info"]
        })
    results_df.append({
        "Configuration": config,
        "Metric": "Entropy",
        "Type": "Entropy",
        "Composite Score": results["entropy_score"],
        "Variance": results["variance"],
        "Mutual Info": results["entropy_mi"],
        "Silhouette": None,
        "Davies-Bouldin": None
    })
    results_df.append({
        "Configuration": config,
        "Metric": "j_mean",
        "Type": "Univariate",
        "Value": results["j_stats"]["mean"]
    })
    results_df.append({
        "Configuration": config,
        "Metric": "j_variance",
        "Type": "Univariate",
        "Value": results["j_stats"]["variance"]
    })
    results_df.append({
        "Configuration": config,
        "Metric": "theta_mean",
        "Type": "Univariate",
        "Value": results["theta_stats"]["mean"]
    })
    results_df.append({
        "Configuration": config,
        "Metric": "j_theta_correlation",
        "Type": "Multivariate",
        "Value": results["multi_stats"]["correlation"]
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
    print(f"Rank {rank}: {config} (Best Divergence: {best_div}, Score: {score:.4f})")

entropy_ranking = sorted(
    [(config, results["entropy_score"]) 
     for config, results in best_measures.items()],
    key=lambda x: x[1],
    reverse=True
)
print("\nEntropy Ranking:")
for rank, (config, score) in enumerate(entropy_ranking, 1):
    print(f"Rank {rank}: {config} (Entropy Score: {score:.4f})")
