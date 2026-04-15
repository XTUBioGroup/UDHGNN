# utils.py
import random
import torch
import numpy as np
import pandas as pd
from dhg import Hypergraph
import scipy.sparse as sp
import os
import math


def load_msigdb_incidence_matrix(data_path):
    """
    Build gene-gene set incidence matrix from MSigDB database
    """
    # Load gene list from MSigDB
    msigdb_path = './Data/msigdb/'
    msigdb_genelist = pd.read_csv(os.path.join(msigdb_path, 'geneList.csv'), header=None)
    msigdb_genelist = list(msigdb_genelist[0].values)

    # Load gene names from dataset
    feature_genename_file = f'{data_path}/feature_genename.txt'
    filtered_geneList = pd.read_csv(feature_genename_file, header=None).iloc[:, 0].tolist()

    # Build incidence matrix
    ids = ['c2', 'c5']
    incidence_matrix = pd.DataFrame(index=filtered_geneList)

    for id in ids:
        # Load gene set names
        geneSetNameList = pd.read_csv(os.path.join(msigdb_path, f'{id}Name.txt'), sep='\t', header=None)
        geneSetNameList = list(geneSetNameList[0].values)

        # Filter cancer-related gene sets
        idList = []
        for z, name in enumerate(geneSetNameList):
            if id == 'c2':
                q = name.split('_')
                cancer_terms = ['CANCER', 'TUMOR', 'NEOPLASM', 'CARCINOMA', 'LEUKEMIA', 'SARCOMA']
                if not any(term in q for term in cancer_terms):
                    idList.append(z)
            elif name[:2] == 'HP':
                q = name.split('_')
                cancer_terms = ['CANCER', 'TUMOR', 'NEOPLASM', 'CARCINOMA', 'LEUKEMIA', 'SARCOMA']
                if not any(term in q for term in cancer_terms):
                    idList.append(z)
            else:
                idList.append(z)

        # Load gene set matrix
        genesetData = sp.load_npz(os.path.join(msigdb_path, f'{id}_GenesetsMatrix.npz'))
        incidence_matrix_temp = pd.DataFrame(data=genesetData.A, index=msigdb_genelist)

        # Filter to dataset genes
        incidence_matrix_temp = incidence_matrix_temp.reindex(index=filtered_geneList, fill_value=0)
        incidence_matrix_temp = incidence_matrix_temp.iloc[:, idList]

        # Concatenate to total incidence matrix
        incidence_matrix = pd.concat([incidence_matrix, incidence_matrix_temp], axis=1)

    # Reset column indices
    incidence_matrix.columns = range(incidence_matrix.shape[1])

    return incidence_matrix


def adj_to_edge_index(adj):
    """
    Convert adjacency matrix to edge index format
    Supports multiple formats: sparse tensor, dense tensor, sparse matrix, dense matrix
    """
    if torch.is_tensor(adj):
        # Handle PyTorch tensor
        if adj.is_sparse:
            # Handle sparse tensor
            adj_coo = adj.coalesce()
            indices = adj_coo.indices().cpu().numpy()
            edge_index = torch.tensor(indices, dtype=torch.long)
        else:
            # Handle dense tensor
            adj_np = adj.cpu().numpy()
            if sp.issparse(adj_np):
                # If it's a scipy sparse matrix format
                adj_coo = adj_np.tocoo()
                edge_index = torch.tensor(
                    np.vstack([adj_coo.row, adj_coo.col]),
                    dtype=torch.long
                )
            else:
                # Dense matrix
                indices = np.where(adj_np > 0)
                edge_index = torch.tensor(
                    np.vstack(indices),
                    dtype=torch.long
                )
    elif sp.issparse(adj):
        # Handle scipy sparse matrix
        adj_coo = adj.tocoo()
        edge_index = torch.tensor(
            np.vstack([adj_coo.row, adj_coo.col]),
            dtype=torch.long
        )
    else:
        # Handle numpy array
        adj_np = np.array(adj)
        indices = np.where(adj_np > 0)
        edge_index = torch.tensor(
            np.vstack(indices),
            dtype=torch.long
        )

    return edge_index


def build_weighted_hypergraph(positive_genes, gene_list, incidence_matrix):
    """
    Build weighted hypergraph adjacency matrix (based on the original model's hypergraph construction)

    Args:
        positive_genes: List of positive gene indices
        gene_list: Complete list of all genes
        incidence_matrix: Pathway-gene association matrix

    Returns:
        G: Hypergraph Laplacian matrix [num_nodes, num_nodes]
    """
    if len(positive_genes) == 0:
        return None

    # Ensure positive genes are present in the incidence matrix
    valid_positive_genes = [g for g in positive_genes if g in incidence_matrix.index]

    if len(valid_positive_genes) == 0:
        return None

    # Count number of positive genes in each pathway
    positive_matrix_sum = incidence_matrix.loc[valid_positive_genes].sum()

    # Select pathways containing at least 3 positive genes (original model threshold)
    sel_hyperedge_idx = np.where(positive_matrix_sum >= 3)[0]

    if len(sel_hyperedge_idx) == 0:
        return None

    # Select relevant pathways
    sel_hyperedge = incidence_matrix.iloc[:, sel_hyperedge_idx]

    # Compute hyperedge weights (original model weight calculation)
    hyperedge_weight = positive_matrix_sum[sel_hyperedge_idx].values
    sel_hyperedge_weight_sum = incidence_matrix.iloc[:, sel_hyperedge_idx].values.sum(0)
    hyperedge_weight = hyperedge_weight / (sel_hyperedge_weight_sum + 1e-10)

    # Build incidence matrix H
    H = np.array(sel_hyperedge).astype('float')

    # Add random perturbation to prevent zero degree (original model handling)
    DV = np.sum(H * hyperedge_weight, axis=1)
    for i in range(DV.shape[0]):
        if DV[i] == 0:
            t = random.randint(0, H.shape[1] - 1)
            H[i][t] = 0.0001

    # Generate hypergraph adjacency matrix G (original model method)
    G = _generate_G_from_H_weight(H, hyperedge_weight)

    return torch.tensor(G, dtype=torch.float32)


def _generate_G_from_H_weight(H, W):
    """
    Generate hypergraph adjacency matrix from weighted incidence matrix (original model method)
    """
    n_edge = H.shape[1]
    DV = np.sum(H * W, axis=1)  # Node degree
    DE = np.sum(H, axis=0)  # Hyperedge degree

    # Handle zero degrees
    DV[DV == 0] = 1e-10
    DE[DE == 0] = 1e-10

    invDE = np.mat(np.diag(1 / DE))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W_mat = np.mat(np.diag(W))
    H_mat = np.mat(H)
    HT = H_mat.T

    G = DV2 * H_mat * W_mat * invDE * HT * DV2
    return G


def load_kfold_data(fold_path, device):
    """
    Load k-fold cross-validation data
    """
    # Load index files
    train_idx = np.loadtxt(f"{fold_path}/train.txt", dtype=int)
    valid_idx = np.loadtxt(f"{fold_path}/valid.txt", dtype=int)
    test_idx = np.loadtxt(f"{fold_path}/test.txt", dtype=int)

    # Load label file
    labels = np.loadtxt(f"{fold_path}/labels.txt", dtype=int)

    # Convert to PyTorch tensors
    train_idx = torch.LongTensor(train_idx).to(device)
    valid_idx = torch.LongTensor(valid_idx).to(device)
    test_idx = torch.LongTensor(test_idx).to(device)

    labels = torch.FloatTensor(labels).to(device)

    return train_idx, valid_idx, test_idx, labels


def check_fold_exists(fold_path):
    """Check if fold folder exists and contains required files"""
    if not os.path.exists(fold_path):
        return False

    required_files = ['labels.txt', 'train.txt', 'valid.txt', 'test.txt']

    for file in required_files:
        if not os.path.exists(f"{fold_path}/{file}"):
            return False

    return True


def load_label_single(path, cancerType, device):
    """
    Load label data for a specific cancer (same as GRAFT)

    Args:
        path: Data path
        cancerType: Cancer type
        device: Device

    Returns:
        Y: Label vector [num_genes]
        label_pos: List of positive gene indices
        label_neg: List of negative gene indices
    """
    import torch
    import numpy as np

    # Load full label vector
    label_file = f"{path}/label_file-P-{cancerType}.txt"
    label = np.loadtxt(label_file)
    Y = torch.tensor(label, dtype=torch.float32, device=device)

    # Load positive gene indices
    pos_file = f"{path}/pos-{cancerType}.txt"
    label_pos = np.loadtxt(pos_file, dtype=int)

    # Load negative gene indices (using pan-cancer negative samples)
    neg_file = f"{path}/pan-neg.txt"
    label_neg = np.loadtxt(neg_file, dtype=int)

    return Y, label_pos, label_neg


def stratified_kfold_split(pos_label, neg_label, l, l1, l2):
    """
    Stratified k-fold split (same as GRAFT)

    Args:
        pos_label: List of positive gene indices
        neg_label: List of negative gene indices
        l: Total number of genes
        l1: Number of positive genes per fold
        l2: Number of negative genes per fold

    Returns:
        List of tuples (train_idx, val_idx, test_idx, train_mask, val_mask, test_mask)
    """
    folds = []

    for i in range(10):
        # 1. Test set split
        pos_test = list(pos_label[i * l1:(i + 1) * l1])
        pos_train = list(set(pos_label) - set(pos_test))
        neg_test = list(neg_label[i * l2:(i + 1) * l2])
        neg_train = list(set(neg_label) - set(neg_test))

        # 2. Split validation set from training set (about 1/8)
        val_size_pos = len(pos_train) // 8
        val_size_neg = len(neg_train) // 8

        pos_val = list(pos_train[:val_size_pos])
        pos_train_final = list(pos_train[val_size_pos:])
        neg_val = list(neg_train[:val_size_neg])
        neg_train_final = list(neg_train[val_size_neg:])

        # 3. Merge indices
        train_idx = sorted(pos_train_final + neg_train_final)
        val_idx = sorted(pos_val + neg_val)
        test_idx = sorted(pos_test + neg_test)

        # 4. Create boolean masks (same as GRAFT)
        indexs1 = [False] * l
        indexs2 = [False] * l
        indexs3 = [False] * l

        for j in train_idx:
            if j < l:  # Ensure index within range
                indexs1[j] = True

        for j in val_idx:
            if j < l:
                indexs2[j] = True

        for j in test_idx:
            if j < l:
                indexs3[j] = True

        train_mask = torch.from_numpy(np.array(indexs1))
        val_mask = torch.from_numpy(np.array(indexs2))
        test_mask = torch.from_numpy(np.array(indexs3))

        folds.append((train_idx, val_idx, test_idx, train_mask, val_mask, test_mask))

    return folds


def check_specific_cancer_data_exists(data_path, cancer_type):
    """
    Check if data for a specific cancer exists (same as GRAFT)

    Args:
        data_path: Data path
        cancer_type: Cancer type

    Returns:
        bool: Whether data exists
    """
    import os

    specific_cancer_path = f"{data_path}/dataset/specific-cancer/"
    required_files = [
        f"label_file-P-{cancer_type}.txt",
        f"pos-{cancer_type}.txt",
        "pan-neg.txt"
    ]

    for file in required_files:
        file_path = os.path.join(specific_cancer_path, file)
        if not os.path.exists(file_path):
            print(f"Warning: File does not exist: {file_path}")
            return False

    return True