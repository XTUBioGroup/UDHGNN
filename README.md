# **Hybrid Graph Convolutional Networks with Directed, Undirected, and Hypergraph Structures for Cancer Driver Gene Identification**



### **File Structure & Description**

Core Code Files

```
main.py
The main entry script of the project. It parses command-line parameters (dataset type, cancer type), loads datasets, initializes
the UDHGNN model, controls the entire training, validation, and testing process, and outputs final prediction results.

model.py
Defines the core architecture of the Hybrid Graph Convolutional Network. It integrates directed graph convolution, undirected
graph convolution, and hypergraph convolution modules, and implements the forward propagation logic of the model.

evaluate.py
Implements model evaluation functions, including calculating classification metrics (AUPRC, F1-score, AUROC).

utils.py
Collects general utility functions, including data preprocessing, data loading, file path management, random seed setting, log 
recording, and auxiliary calculation functions.
```



### **Data Directory** 

The core dataset folder stores all biological network data, gene annotation files, cancer driver gene labels, and reference databases required for model training and testing.

```
Sub-files & Sub-directories
Gene Label Files
  796true.txt: Stores positive samples (796 validated cancer driver genes, standard positive labels).
  2187false.txt: Stores negative samples (2187 non-cancer driver genes, standard negative labels).
These files provide supervised learning labels for model training.

CPDB/
The core PPI network under the CPDB directory is sourced from the ConsensusPathDB database, which aggregates high-confidence
human protein interaction information from multiple mainstream interaction databases. The node sets of the auxiliary networks 
within this directory—the Pathway functional similarity network, the GO semantic similarity network, the RegNetwork gene 
regulatory network, and the gene-set hypergraph based on MSigDB—have likewise been aligned with the gene nodes in the CPDB PPI network.

STRING/
The core protein–protein interaction (PPI) network under the STRING directory is sourced from the STRING database (v11.5), which
integrates diverse evidence types—including experimental validation, co-expression analysis, and text mining—to provide weighted
interaction relationships among human genes. The auxiliary networks within this directory—namely the Pathway functional 
similarity network, the GO semantic similarity network, the RegNetwork gene regulatory network, and the gene-set hypergraph 
constructed from MSigDB—have all had their node sets strictly aligned with the gene nodes in the STRING PPI network, ensuring consistent node space across the multi-view graph learning process.

msigdb/
Molecular Signatures Database. Stores gene set annotation data, including hallmark gene sets, oncogenic signatures, and pathway-
related gene sets. Used for gene feature enhancement, functional enrichment analysis, and auxiliary model feature learning.
```



### **Dependence on installation：**

```
# 1. Create and activate conda environment (Python 3.9)
conda create -n udhgcn_env python=3.9 -y
conda activate udhgcn_env

# 2. Install PyTorch 1.13.1 + CUDA 11.7 (using pip)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# 3. Install scipy and other scientific computing libraries (to provide dependencies for PyG)
pip install numpy==1.26.4 scipy==1.13.1 pandas==2.3.3 scikit-learn==1.6.1

# 4. Install PyG core dependencies (force precompiled wheels from official PyG link to avoid source compilation)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    --find-links https://data.pyg.org/whl/torch-1.13.1+cu117.html \
    --no-index

# 5. Install PyTorch Geometric 2.6.1
pip install torch-geometric==2.6.1

# 6. Install dhg (hypergraph library)
pip install dhg==0.9.5

# 7. (Optional) Verify installation
python -c "import torch; import torch_geometric; import dhg; print('All OK')"
```



### Example Usage：

```
# Run UDHGNN using STRING network for pan-cancer prediction
python main.py --dataset STRING --cancer_type pan-cancer

# Run UDHGNN using CPDB network for kirp cancer type
python main.py --dataset CPDB --cancer_type kirp
```

