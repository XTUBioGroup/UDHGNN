# main.py
import torch
import numpy as np
import pandas as pd
import argparse
import random
import os
import gc
import time
from sklearn.preprocessing import StandardScaler

# Custom modules
from utils import (
    load_kfold_data, check_fold_exists, adj_to_edge_index,
    build_weighted_hypergraph, stratified_kfold_split, load_label_single
)
from model import UDHGNN
from evaluate import evaluate_predictions

# Argument parsing
parser = argparse.ArgumentParser(description='UDHGNN Cancer Driver Gene Identification Model')
parser.add_argument('--dataset', type=str, default='CPDB', help='Dataset name')
parser.add_argument('--cancer_type', type=str, default='pan-cancer', help='Cancer type')
parser.add_argument('--hidden_dim', type=int, default=32, help='Multi-view hidden dimension')
parser.add_argument('--hyper_hidden_dim', type=int, default=256, help='Hypergraph encoder hidden dimension')
parser.add_argument('--directed_hidden_dim', type=int, default=256, help='Directed graph encoder hidden dimension')
parser.add_argument('--seed', type=int, default=1234, help='Random seed')
parser.add_argument('--device', type=int, default=0, help='GPU device ID')
parser.add_argument('--epochs', type=int, default=600, help='Number of epochs')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.8, help='Focal Loss alpha parameter')
parser.add_argument('--gamma', type=float, default=2.5, help='Focal Loss gamma parameter')
parser.add_argument('--heads', type=int, default=4, help='Number of attention heads in GATGCN')
parser.add_argument('--undirected_layers', type=int, default=3, help='Number of undirected graph encoder layers')
parser.add_argument('--directed_layers', type=int, default=3, help='Number of directed graph encoder layers')
parser.add_argument('--hyper_layers', type=int, default=3, help='Number of hypergraph encoder layers')
args = parser.parse_args()

# Set random seed and device
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def train_fold(node_features, undirected_edge_indices, directed_edge_index, labels,
               train_idx, valid_idx, test_idx, incidence_matrix,
               args, device, fold_num):
    """
    Train a single fold
    """
    print(f"\n=== Training Fold {fold_num} ===")
    start_time = time.time()

    num_nodes = node_features.shape[0]
    feature_dim = node_features.shape[1]

    # Build hypergraph adjacency matrix G
    hypergraph_G = None
    if incidence_matrix is not None:
        train_labels = labels[train_idx]
        positive_idx = train_idx[train_labels == 1].cpu().numpy().tolist()
        gene_list = list(range(num_nodes))

        if not all(i in incidence_matrix.index for i in positive_idx):
            incidence_matrix_reset = incidence_matrix.reset_index(drop=True)
        else:
            incidence_matrix_reset = incidence_matrix

        hypergraph_G = build_weighted_hypergraph(
            positive_idx, gene_list, incidence_matrix_reset
        )

        if hypergraph_G is not None:
            hypergraph_G = hypergraph_G.to(device)
        else:
            print("Warning: Hypergraph construction failed, hypergraph features will be omitted.")
            # In this case we will still proceed, but hypergraph will be None

    # Create model
    model = UDHGNN(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_undirected_networks=len(undirected_edge_indices),  # Use the number of loaded networks
        dropout=args.dropout,
        alpha=args.alpha,
        gamma=args.gamma,
        gat_heads=args.heads,
        directed_hidden_dim=args.directed_hidden_dim,
        hyper_hidden_dim=args.hyper_hidden_dim,
        undirected_layers=args.undirected_layers,
        directed_layers=args.directed_layers,
        hyper_layers=args.hyper_layers
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        if directed_edge_index is not None:
            outputs = model(node_features, undirected_edge_indices, directed_edge_index, hypergraph_G)
        else:
            outputs = model(node_features, undirected_edge_indices, None, hypergraph_G)

        # Compute loss
        loss = model.compute_loss(
            outputs, labels.unsqueeze(1),
            mask=train_idx
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if directed_edge_index is not None:
                val_outputs = model(node_features, undirected_edge_indices, directed_edge_index, hypergraph_G)
            else:
                val_outputs = model(node_features, undirected_edge_indices, None, hypergraph_G)
            val_probs = torch.sigmoid(val_outputs[valid_idx]).cpu().numpy()
            val_labels = labels[valid_idx].cpu().numpy()

            # Compute validation metrics
            from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
            import numpy as np

            try:
                val_auroc = roc_auc_score(val_labels, val_probs)
            except:
                val_auroc = 0.5

            try:
                val_auprc = average_precision_score(val_labels, val_probs)
            except:
                val_auprc = float(np.mean(val_labels))

            try:
                val_preds = (val_probs > 0.5).astype(int)
                val_f1 = f1_score(val_labels, val_preds, average='binary')
            except:
                val_f1 = 0.0

        # Print progress
        if epoch % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                  f"Val AUROC={val_auroc:.4f}, Val AUPRC={val_auprc:.4f}, Val F1={val_f1:.4f}")

    # Test evaluation
    model.eval()
    with torch.no_grad():
        if directed_edge_index is not None:
            test_outputs = model(node_features, undirected_edge_indices, directed_edge_index, hypergraph_G)
        else:
            test_outputs = model(node_features, undirected_edge_indices, None, hypergraph_G)
        auroc, auprc, f1 = evaluate_predictions(
            test_outputs, labels, test_idx
        )

    training_time = time.time() - start_time
    print(f"\nFold {fold_num} Test Results: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, F1={f1:.4f}")
    print(f"Training time: {training_time:.2f}s")

    return auroc, auprc, f1


def main():
    print(f"===== UDHGNN Cancer Driver Gene Identification Model (Integrated Directed Graph Encoder) =====")
    print(f"Dataset: {args.dataset}, Cancer Type: {args.cancer_type}")
    print(f"Device: {device}, Random seed: {args.seed}")
    print(f"GAT attention heads: {args.heads}")
    print(f"Undirected layers: {args.undirected_layers}, Directed layers: {args.directed_layers}, Hypergraph layers: {args.hyper_layers}")

    # 1. Load data
    print("\n[1/4] Loading data...")
    data_path = f"./Data/{args.dataset}"
    feature_file = f"{data_path}/multiomics_features_{args.dataset}.tsv"

    if not os.path.exists(feature_file):
        print(f"Feature file does not exist: {feature_file}")
        return

    data_x_df = pd.read_csv(feature_file, sep='\t', index_col=0)
    data_x_df = data_x_df.dropna()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data_x_df.values)
    node_features = torch.tensor(features_scaled, dtype=torch.float32, device=device)

    # Select features based on cancer type
    cancer_type = args.cancer_type.lower()
    if cancer_type == 'pan-cancer':
        node_features = node_features[:, :48]
        print("Using pan-cancer features (48 dimensions)")
    else:
        cancer_type_dict = {
            'kirc': [0, 16, 32],
            'brca': [1, 17, 33],
            'prad': [3, 19, 35],
            'stad': [4, 20, 36],
            'hnsc': [5, 21, 37],
            'luad': [6, 22, 38],
            'thca': [7, 23, 39],
            'blca': [8, 24, 40],
            'esca': [9, 25, 41],
            'lihc': [10, 26, 42],
            'ucec': [11, 27, 43],
            'coad': [12, 28, 44],
            'lusc': [13, 29, 45],
            'cesc': [14, 30, 46],
            'kirp': [15, 31, 47]
        }
        if cancer_type in cancer_type_dict:
            node_features = node_features[:, cancer_type_dict[cancer_type]]
            print(f"Using {cancer_type.upper()} cancer features ({len(cancer_type_dict[cancer_type])} dimensions)")
        else:
            print(f"Warning: No configuration for cancer type {cancer_type}, using first 3 features")
            node_features = node_features[:, :3]

    print(f"Node feature shape: {node_features.shape}")
    num_nodes = node_features.shape[0]

    # 2. Load graph structures
    print("\n[2/4] Loading graph structures...")

    # Load undirected graphs (PPI, Pathway, GO)
    undirected_graph_files = {
        'ppi': f"{data_path}/{args.dataset}_ppi.pkl",
        'pathway': f"{data_path}/pathway_SimMatrix_filtered.pkl",
        'go': f"{data_path}/GO_SimMatrix_filtered.pkl"
    }

    undirected_edge_indices = []
    for name, file_path in undirected_graph_files.items():
        if os.path.exists(file_path):
            try:
                adj = torch.load(file_path)
                edge_index = adj_to_edge_index(adj).to(device)
                undirected_edge_indices.append(edge_index)
                print(f"  Loaded {name} network (undirected), edges: {edge_index.shape[1]}")
            except Exception as e:
                print(f"  Failed to load {name} network: {e}")
        else:
            print(f"  Warning: {name} network file does not exist: {file_path}")

    if len(undirected_edge_indices) == 0:
        print("Error: No undirected graph network loaded")
        return

    print(f"Loaded {len(undirected_edge_indices)} undirected networks")

    # Load directed graph (fixed using regnet)
    directed_edge_index = None
    directed_graph_file = f"{data_path}/RegNetwork_full_adj.pkl"
    if os.path.exists(directed_graph_file):
        try:
            print(f"Loading directed graph (regnet): {directed_graph_file}")
            adj = torch.load(directed_graph_file)
            directed_edge_index = adj_to_edge_index(adj).to(device)
            print(f"  Directed graph edges: {directed_edge_index.shape[1]}")

            # Check for self-loops
            src, dst = directed_edge_index
            self_loop_count = (src == dst).sum().item()
            if self_loop_count > 0:
                print(f"  Detected {self_loop_count} self-loop edges in directed graph")
        except Exception as e:
            print(f"  Failed to load directed graph: {e}")
            print("  Exiting because directed graph is required.")
            return
    else:
        print(f"  Error: Directed graph file does not exist: {directed_graph_file}")
        print("  Exiting because directed graph is required.")
        return

    # 3. Load incidence matrix (for hypergraph)
    print("\n[3/4] Loading incidence matrix...")
    incidence_matrix = None
    try:
        from utils import load_msigdb_incidence_matrix
        incidence_matrix = load_msigdb_incidence_matrix(data_path)
        print(f"Incidence matrix shape: {incidence_matrix.shape}")
    except Exception as e:
        print(f"Incidence matrix construction failed: {e}")
        print("  Exiting because hypergraph is required.")
        return

    # 4. Training and evaluation
    print("\n[4/4] Training and evaluation...")

    if cancer_type == 'pan-cancer':
        print(f"Node feature dimension: {node_features.shape[1]}")
        print(f"Number of undirected networks: {len(undirected_edge_indices)}")
        print(f"Undirected hidden dimension: {args.hidden_dim}")
        print(f"Directed hidden dimension: {args.directed_hidden_dim}")
        print(f"Hypergraph hidden dimension: {args.hyper_hidden_dim}")
        print(f"GAT attention heads: {args.heads}")
        print(f"Undirected layers: {args.undirected_layers}")
        print(f"Directed layers: {args.directed_layers}")
        print(f"Hypergraph layers: {args.hyper_layers}")
        print(f"Dropout rate: {args.dropout}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Focal Loss alpha: {args.alpha}")
        print(f"Focal Loss gamma: {args.gamma}")
        print("=" * 60 + "\n")

        # Pan-cancer: use pre-generated 10-fold
        all_aurocs, all_auprcs, all_f1s = [], [], []

        for fold_num in range(1, 11):
            fold_path = f"{data_path}/10fold/fold_{fold_num}"

            if not check_fold_exists(fold_path):
                print(f"Fold {fold_num} does not exist, skipping")
                continue

            try:
                # Load fold data
                train_idx, valid_idx, test_idx, labels = load_kfold_data(
                    fold_path, device
                )

                # Adjust label shape
                if labels.shape[0] != num_nodes:
                    new_labels = torch.zeros(num_nodes, device=device)
                    new_labels[:labels.shape[0]] = labels
                    labels = new_labels

                print(f"\nFold {fold_num}: Train={len(train_idx)}, Valid={len(valid_idx)}, Test={len(test_idx)}")
                print(f"Positive samples: {int(labels.sum().item())}, Negative samples: {len(train_idx)+len(valid_idx)+len(test_idx) - int(labels.sum().item())}")

                # Train
                auroc, auprc, f1 = train_fold(
                    node_features, undirected_edge_indices, directed_edge_index, labels,
                    train_idx, valid_idx, test_idx, incidence_matrix,
                    args, device, fold_num
                )

                all_aurocs.append(auroc)
                all_auprcs.append(auprc)
                all_f1s.append(f1)

                # Clean memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Fold {fold_num} training failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Output results
        if all_aurocs:
            print("\n" + "=" * 60)
            print("Cross-validation results:")
            print("=" * 60)
            print(f"Average AUROC:  {np.mean(all_aurocs):.4f} ± {np.std(all_aurocs):.4f}")
            print(f"Average AUPRC:  {np.mean(all_auprcs):.4f} ± {np.std(all_auprcs):.4f}")
            print(f"Average F1:     {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")

            # Save results
            result_dir = './results'
            os.makedirs(result_dir, exist_ok=True)

            results = {
                'auroc': {'mean': np.mean(all_aurocs), 'std': np.std(all_aurocs), 'all': all_aurocs},
                'auprc': {'mean': np.mean(all_auprcs), 'std': np.std(all_auprcs), 'all': all_auprcs},
                'f1': {'mean': np.mean(all_f1s), 'std': np.std(all_f1s), 'all': all_f1s},
                'args': vars(args)
            }

            result_file = f"{result_dir}/{args.dataset}_{args.cancer_type}_with_directed_results.npy"
            np.save(result_file, results, allow_pickle=True)
            print(f"Results saved to: {result_file}")
        else:
            print("No valid training results obtained")

    else:
        # Specific cancer: use the same data split strategy as GRAFT
        print(f"Specific cancer type {cancer_type}")

        # Load labels for this cancer type
        specific_cancer_path = f"{data_path}/dataset/specific-cancer/"
        try:
            labels, label_pos, label_neg = load_label_single(
                specific_cancer_path, cancer_type, device
            )
        except Exception as e:
            print(f"Failed to load specific cancer data: {e}")
            import traceback
            traceback.print_exc()
            return

        # Ensure labels match number of nodes
        if labels.shape[0] != num_nodes:
            print(f"Warning: Label count ({labels.shape[0]}) does not match node count ({num_nodes}), adjusting...")
            if labels.shape[0] > num_nodes:
                labels = labels[:num_nodes]
            else:
                new_labels = torch.zeros(num_nodes, device=device)
                new_labels[:labels.shape[0]] = labels
                labels = new_labels

        # ===== AUTO-HYPERPARAMETER SELECTION BASED ON POSITIVE SAMPLE COUNT =====
        pos_count = len(label_pos)
        print(f"Number of positive samples in dataset: {pos_count}")

        if pos_count < 30:
            print("Using hyperparameters for small positive sample dataset (pos < 30)")
            args.undirected_layers = 3
            args.directed_layers = 3
            args.hyper_layers = 2
            args.heads = 2
            args.hidden_dim = 64
            args.directed_hidden_dim = 64
            args.hyper_hidden_dim = 32
            args.epochs = 400
            args.lr = 20e-4
            args.weight_decay = 3e-5
            args.dropout = 0.3
            args.alpha = 0.90
            args.gamma = 5
        else:
            print("Using default hyperparameters (pos >= 30)")
            args.undirected_layers = 3
            args.directed_layers = 3
            args.hyper_layers = 3
            args.heads = 2
            args.hidden_dim = 256
            args.directed_hidden_dim = 256
            args.hyper_hidden_dim = 32
            args.epochs = 400
            args.lr = 25e-4
            args.weight_decay = 1e-5
            args.dropout = 0.2
            args.alpha = 0.9
            args.gamma = 3

        # Print final hyperparameters
        print("\n" + "=" * 60)
        print("Model hyperparameter configuration:")
        print(f"Node feature dimension: {node_features.shape[1]}")
        print(f"Number of undirected networks: {len(undirected_edge_indices)}")
        print(f"Undirected hidden dimension: {args.hidden_dim}")
        print(f"Directed hidden dimension: {args.directed_hidden_dim}")
        print(f"Hypergraph hidden dimension: {args.hyper_hidden_dim}")
        print(f"GAT attention heads: {args.heads}")
        print(f"Undirected layers: {args.undirected_layers}")
        print(f"Directed layers: {args.directed_layers}")
        print(f"Hypergraph layers: {args.hyper_layers}")
        print(f"Dropout rate: {args.dropout}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Focal Loss alpha: {args.alpha}")
        print(f"Focal Loss gamma: {args.gamma}")
        print("=" * 60 + "\n")

        # Use the same stratified split strategy as GRAFT
        total_genes = labels.shape[0]
        pos_per_fold = len(label_pos) // 10
        neg_per_fold = len(label_neg) // 10

        folds = stratified_kfold_split(
            label_pos, label_neg,
            total_genes, pos_per_fold, neg_per_fold
        )

        all_aurocs, all_auprcs, all_f1s = [], [], []

        for i, (train_idx, valid_idx, test_idx, train_mask, val_mask, test_mask) in enumerate(folds):

            print(f"\nFold {i + 1}: Train={len(train_idx)}, Valid={len(valid_idx)}, Test={len(test_idx)}")
            print(f"Positive samples: {int(labels.sum().item())}, Negative samples: {len(train_idx)+len(valid_idx)+len(test_idx) - int(labels.sum().item())}")
            # Convert to Tensors
            train_idx = torch.LongTensor(train_idx).to(device)
            valid_idx = torch.LongTensor(valid_idx).to(device)
            test_idx = torch.LongTensor(test_idx).to(device)

            # Train
            auroc, auprc, f1 = train_fold(
                node_features, undirected_edge_indices, directed_edge_index, labels,
                train_idx, valid_idx, test_idx, incidence_matrix,
                args, device, i + 1
            )

            all_aurocs.append(auroc)
            all_auprcs.append(auprc)
            all_f1s.append(f1)

            # Clean memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Output results
        if all_aurocs:
            print("\n" + "=" * 60)
            print(f"{cancer_type.upper()} Cancer Results:")
            print("=" * 60)
            print(f"Average AUROC:  {np.mean(all_aurocs):.4f} ± {np.std(all_aurocs):.4f}")
            print(f"Average AUPRC:  {np.mean(all_auprcs):.4f} ± {np.std(all_auprcs):.4f}")
            print(f"Average F1:     {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")

            # Save results
            result_dir = './results'
            os.makedirs(result_dir, exist_ok=True)

            results = {
                'auroc': {'mean': np.mean(all_aurocs), 'std': np.std(all_aurocs), 'all': all_aurocs},
                'auprc': {'mean': np.mean(all_auprcs), 'std': np.std(all_auprcs), 'all': all_auprcs},
                'f1': {'mean': np.mean(all_f1s), 'std': np.std(all_f1s), 'all': all_f1s},
                'args': vars(args)
            }

            result_file = f"{result_dir}/{args.dataset}_{args.cancer_type}_with_directed_results.npy"
            np.save(result_file, results, allow_pickle=True)
            print(f"Results saved to: {result_file}")
        else:
            print("No valid training results obtained")

    print("\n===== Analysis completed =====")


if __name__ == "__main__":
    main()