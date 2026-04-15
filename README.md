Hybrid Graph Convolutional Networks with Directed, Undirected, and Hypergraph Structures for Cancer Driver Gene Identification



Dependence on installation：

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