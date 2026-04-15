# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv, GATConv
from torch_geometric.utils import dropout_edge
from torch.nn.parameter import Parameter
import math


class HGNN_conv(nn.Module):
    """Hypergraph convolution layer (original model HGNN_conv)"""

    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HypergraphHGNN(nn.Module):
    """Hypergraph neural network (original model hypergrph_HGNN), supports configurable layers"""

    def __init__(self, num_nodes, n_hid, dropout=0.5, num_layers=3):
        super(HypergraphHGNN, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.num_layers = num_layers

        self.fc = nn.Linear(num_nodes, n_hid)

        # Create multiple HGNN_conv layers
        self.hgnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.hgnn_layers.append(HGNN_conv(n_hid, n_hid))

    def forward(self, G):
        x = torch.eye(self.num_nodes, device=G.device).float()
        x = F.relu(self.fc(x))
        x = F.dropout(x, self.dropout, training=self.training)

        # Multi-layer HGNN convolution
        for i in range(self.num_layers):
            x_res = x  # Residual connection
            x = F.relu(self.hgnn_layers[i](x, G) + x_res)
            if i < self.num_layers - 1:  # No dropout after last layer
                x = F.dropout(x, self.dropout, training=self.training)

        return x


class Attention(nn.Module):
    """Attention mechanism for aggregating embeddings from multiple graph views"""

    def __init__(self, nb_graphs, hid_units):
        super(Attention, self).__init__()
        self.nb_graphs = nb_graphs
        self.hid_units = hid_units
        self.attention_weights = nn.Parameter(torch.ones(nb_graphs) / nb_graphs)

    def forward(self, features_list):
        weights = F.softmax(self.attention_weights, dim=0)
        aggregated_features = torch.zeros_like(features_list[0])
        for i in range(self.nb_graphs):
            aggregated_features += weights[i] * features_list[i]
        return aggregated_features, weights


class NetworkFeatureExtractor(nn.Module):
    """
    Network feature extractor: integrates the gcn_mlp method from SGL
    Extracts specific features for each network (used for undirected graphs), supports configurable layers
    """

    def __init__(self, input_dim, hidden_dim, output_dim, drop_p=0.5, drop_edge_p=0.5, num_layers=2):
        super(NetworkFeatureExtractor, self).__init__()
        self.drop_p = drop_p
        self.drop_edge_p = drop_edge_p
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # First layer
        self.fc_first = nn.Linear(input_dim, hidden_dim)
        self.conv_first = GCNConv(input_dim, hidden_dim, add_self_loops=False)

        # If more than 1 layer, create intermediate layers
        if num_layers > 1:
            self.mid_fc_layers = nn.ModuleList()
            self.mid_conv_layers = nn.ModuleList()

            for i in range(num_layers - 1):
                # Intermediate layers: input dimension is hidden_dim * 2
                self.mid_fc_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
                self.mid_conv_layers.append(GCNConv(hidden_dim * 2, hidden_dim, add_self_loops=False))

        # Final layer
        self.final_fc = nn.Linear(hidden_dim * 2, output_dim)
        self.final_conv = GCNConv(hidden_dim * 2, output_dim, add_self_loops=False)

    def forward(self, x, edge_index):
        if self.training:
            edge_index = dropout_edge(
                edge_index,
                p=self.drop_edge_p,
                force_undirected=True,
                training=self.training
            )[0]
            x = F.dropout(x, p=self.drop_p, training=self.training)

        # First layer: use original features
        x1 = torch.relu(self.fc_first(x))
        x2 = torch.relu(self.conv_first(x, edge_index))
        # First layer output: concatenate two branches
        first_layer_output = torch.cat((x1, x2), 1)  # [num_nodes, hidden_dim * 2]

        # Save first layer output for subsequent layers
        if self.num_layers > 1:
            current_output = first_layer_output

        # Second and subsequent layers
        for i in range(self.num_layers - 1):
            if self.training:
                current_output = F.dropout(current_output, p=self.drop_p, training=self.training)

            # For third layer and beyond (i >= 1), add first layer output
            if i >= 1:
                # Add first layer output to current layer input (addition, not concatenation)
                layer_input = current_output + first_layer_output
            else:
                # Second layer: only use first layer output
                layer_input = current_output

            # Two branches
            x1 = torch.relu(self.mid_fc_layers[i](layer_input))
            x2 = torch.relu(self.mid_conv_layers[i](layer_input, edge_index))
            # Output: concatenate two branches
            current_output = torch.cat((x1, x2), 1)  # [num_nodes, hidden_dim * 2]

        # Final layer (if num_layers=1, directly use first layer output)
        if self.num_layers == 1:
            final_input = first_layer_output
        else:
            # For multiple layers, also use the fusion strategy for the final layer
            final_input = current_output

        if self.training:
            final_input = F.dropout(final_input, p=self.drop_p, training=self.training)

        # Last two branches
        x4 = torch.relu(self.final_fc(final_input))
        x5 = torch.relu(self.final_conv(final_input, edge_index))

        # Feature fusion
        embedding = x4 + x5  # [num_nodes, output_dim]

        return embedding


class DirectedNetworkFeatureExtractor(nn.Module):
    """
    Directed graph feature extractor: maintains the same feature dimension changes as the undirected version
    Supports configurable layers and number of attention heads
    Modification: starting from the third layer, add the output of the first layer (addition)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, drop_p=0.5, drop_edge_p=0.5, heads=4, num_layers=2):
        super(DirectedNetworkFeatureExtractor, self).__init__()
        self.drop_p = drop_p
        self.drop_edge_p = drop_edge_p
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim

        # Dimension per attention head
        self.head_dim = hidden_dim // heads

        # Ensure hidden_dim is divisible by heads
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")

        # First layer
        self.fc_first = nn.Linear(input_dim, hidden_dim)
        self.gat_first = GATConv(
            input_dim,
            self.head_dim,
            heads=heads,
            concat=True,
            dropout=drop_p,
            add_self_loops=False
        )

        # If more than 1 layer, create intermediate layers
        if num_layers > 1:
            self.mid_fc_layers = nn.ModuleList()
            self.mid_gat_layers = nn.ModuleList()

            for i in range(num_layers - 1):
                # Intermediate layers
                self.mid_fc_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
                self.mid_gat_layers.append(GATConv(
                    hidden_dim * 2,
                    self.head_dim,
                    heads=heads,
                    concat=True,
                    dropout=drop_p,
                    add_self_loops=False
                ))

        # Final layer
        self.concat_dim = hidden_dim * 2
        self.final_fc = nn.Linear(self.concat_dim, output_dim)
        self.final_gat = GATConv(
            self.concat_dim,
            output_dim,
            heads=1,
            concat=False,
            dropout=drop_p,
            add_self_loops=False
        )

    def forward(self, x, edge_index):
        """Extract directed graph features, add first layer output from third layer onward"""
        if self.training:
            x = F.dropout(x, p=self.drop_p, training=self.training)

        # First layer: use original features
        x1 = torch.relu(self.fc_first(x))  # [num_nodes, hidden_dim]
        x2 = torch.relu(self.gat_first(x, edge_index))  # [num_nodes, hidden_dim]
        # First layer output: concatenate two branches
        first_layer_output = torch.cat((x1, x2), 1)  # [num_nodes, hidden_dim * 2]

        # Save first layer output for subsequent layers
        if self.num_layers > 1:
            current_output = first_layer_output

        # Second and subsequent layers
        for i in range(self.num_layers - 1):
            if self.training:
                current_output = F.dropout(current_output, p=self.drop_p, training=self.training)

            # For third layer and beyond (i >= 1), add first layer output
            if i >= 1:
                # Add first layer output to current layer input (addition, not concatenation)
                layer_input = current_output + first_layer_output
            else:
                # Second layer: only use first layer output
                layer_input = current_output

            # Two branches
            x1 = torch.relu(self.mid_fc_layers[i](layer_input))
            x2 = torch.relu(self.mid_gat_layers[i](layer_input, edge_index))
            # Output: concatenate two branches
            current_output = torch.cat((x1, x2), 1)  # [num_nodes, hidden_dim * 2]

        # Final layer (if num_layers=1, directly use first layer output)
        if self.num_layers == 1:
            final_input = first_layer_output
        else:
            # For multiple layers, also use the fusion strategy for the final layer
            final_input = current_output

        if self.training:
            final_input = F.dropout(final_input, p=self.drop_p, training=self.training)

        # Last two branches
        x4 = torch.relu(self.final_fc(final_input))  # [num_nodes, output_dim]
        x5 = torch.relu(self.final_gat(final_input, edge_index))  # [num_nodes, output_dim]

        # Feature fusion
        embedding = x4 + x5  # [num_nodes, output_dim]

        return embedding


class UDHGNN(nn.Module):
    """
    UDHGNN Classifier: integrates network feature extractors, hypergraph encoder, directed graph encoder, and Chebyshev GCN
    Uses standard Focal Loss
    Supports separate hidden dimensions and layers for hypergraph and directed graph encoders
    """

    def __init__(self, num_nodes, feature_dim, hidden_dim, num_undirected_networks=3,
                 dropout=0.5, alpha=0.25, gamma=2.0, gat_heads=4,
                 directed_hidden_dim=None, hyper_hidden_dim=None,
                 undirected_layers=2, directed_layers=2, hyper_layers=3):
        super(UDHGNN, self).__init__()

        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_undirected_networks = num_undirected_networks
        self.dropout = dropout
        self.alpha = alpha
        self.gamma = gamma
        self.gat_heads = gat_heads
        self.undirected_layers = undirected_layers
        self.directed_layers = directed_layers
        self.hyper_layers = hyper_layers

        # Set directed graph encoder hidden dimension, default to undirected hidden dimension
        if directed_hidden_dim is None:
            directed_hidden_dim = hidden_dim
        self.directed_hidden_dim = directed_hidden_dim

        # Set hypergraph encoder hidden dimension, default to undirected hidden dimension
        if hyper_hidden_dim is None:
            hyper_hidden_dim = hidden_dim
        self.hyper_hidden_dim = hyper_hidden_dim

        # Undirected graph feature extractors (one per undirected network)
        self.undirected_extractors = nn.ModuleList([
            NetworkFeatureExtractor(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,  # Output dimension set to hidden_dim for consistency
                drop_p=dropout,
                drop_edge_p=dropout,
                num_layers=undirected_layers
            ) for _ in range(num_undirected_networks)
        ])

        # Directed graph feature extractor (only one)
        self.directed_extractor = DirectedNetworkFeatureExtractor(
            input_dim=feature_dim,
            hidden_dim=directed_hidden_dim,
            output_dim=directed_hidden_dim,
            drop_p=dropout,
            heads=gat_heads,
            num_layers=directed_layers
        )

        # Attention fusion mechanism (only for undirected graphs)
        self.attention = Attention(num_undirected_networks, hidden_dim)

        # Hypergraph encoder - uses hyper_hidden_dim and layer parameter
        self.hypergraph_encoder = HypergraphHGNN(
            num_nodes=num_nodes,
            n_hid=hyper_hidden_dim,
            dropout=dropout,
            num_layers=hyper_layers
        )

        # Compute total dimension
        total_dim = hidden_dim  # Base: attention fused features
        total_dim += hyper_hidden_dim  # Add hypergraph features
        total_dim += directed_hidden_dim  # Add directed graph features

        # Chebyshev GCN classifier - input dimension is total_dim
        self.cheb_gcn = ChebConv(total_dim, hidden_dim, K=2, normalization='sym')
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        # Print dimension information
        print(f"UDHGNN Model configuration:")
        print(f"  Undirected graph encoder: {num_undirected_networks} networks, {undirected_layers} layers, hidden dimension {hidden_dim}")
        print(f"  Directed graph encoder: {directed_layers} layers, hidden dimension {directed_hidden_dim}, attention heads {gat_heads}")
        print(f"  Hypergraph encoder: {hyper_layers} layers, hidden dimension {hyper_hidden_dim}")
        print(f"  Total feature dimension: {total_dim}")

    def forward(self, node_features, undirected_edge_indices, directed_edge_index=None, hypergraph_G=None):
        """
        Forward pass
        """
        device = node_features.device

        # 1. Extract features from each undirected network
        undirected_features = []
        for i in range(self.num_undirected_networks):
            features = self.undirected_extractors[i](node_features, undirected_edge_indices[i])
            undirected_features.append(features)

        # 2. Attention fusion of undirected graph features
        fused_features, attention_weights = self.attention(undirected_features)

        # 3. Extract directed graph features (always used)
        directed_features = self.directed_extractor(node_features, directed_edge_index)

        # 4. Extract hypergraph features (always used)
        hyper_features = self.hypergraph_encoder(hypergraph_G)

        # 5. Concatenate features
        combined_features = torch.cat([fused_features, directed_features, hyper_features], dim=1)

        # 6. Dropout
        combined_features = F.dropout(combined_features, p=self.dropout, training=self.training)

        # 7. Chebyshev GCN classification (use the first undirected network's edge index)
        cheb_features = self.cheb_gcn(combined_features, undirected_edge_indices[0])
        outputs = self.output_layer(cheb_features)

        return outputs

    def compute_loss(self, outputs, targets, mask=None):
        """Compute standard Focal Loss"""
        if mask is not None:
            outputs = outputs[mask]
            targets = targets[mask]

        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()

        probs = torch.sigmoid(outputs)
        ce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        modulation_factor = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_weight * modulation_factor * ce_loss

        return focal_loss.mean()