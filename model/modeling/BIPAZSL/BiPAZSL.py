import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import os
import pickle
import numpy as np
from os.path import join
from torch.nn.functional import dropout

# ----------------------------
# Utility Functions
# ----------------------------
def trunc_normal_(tensor, mean=0, std=0.01):
    """Truncated normal initialization for tensors."""
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


# ----------------------------
# Attention and Feature Processing Modules
# ----------------------------
class Part_Attention(nn.Module):
    """Module to compute part-level attention weights using cls_token guidance."""
    def __init__(self, num_heads, original_seq_len):
        super(Part_Attention, self).__init__()
        self.num_heads = num_heads
        self.original_seq_len = original_seq_len  # Sequence length including cls_token (e.g., 197)

    def forward(self, hidden_states, attn_weights):
        """
        Args:
            hidden_states: Features from (L-1)-th layer, shape [B, N, C]
            attn_weights: List of attention weights from previous layers, each [B, num_heads, N, N]
        
        Returns:
            Weighted features with shape [B, N, C]
        """
        # Extract attention from cls_token (index 0) to all tokens across layers
        cls_attn_list = []
        for attn in attn_weights:
            # Shape: [B, num_heads, N] (attention from cls_token to all tokens)
            cls_attn = attn[:, :, 0, :]
            cls_attn = cls_attn.mean(dim=1)  # Average over heads -> [B, N]
            cls_attn_list.append(cls_attn)

        # Accumulate and normalize attention weights
        last_map = torch.ones_like(cls_attn_list[0])  # [B, N]
        for attn in cls_attn_list:
            last_map *= attn  # Cumulative importance weighting
        
        last_map = F.softmax(last_map, dim=1)  # Normalize to sum=1 -> [B, N]

        # Apply weights to hidden states
        full_weights = last_map.unsqueeze(-1)  # [B, N, 1] for broadcasting
        weighted_feats = hidden_states * full_weights  # [B, N, C]

        return weighted_feats


class SelfAttention(nn.Module):
    """Standard self-attention module."""
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: Input features, shape [B, N, C]
        
        Returns:
            Attention-weighted features, shape [B, N, C]
        """
        Q = self.query(x)  # [B, N, C]
        K = self.key(x)    # [B, N, C]
        V = self.value(x)  # [B, N, C]

        # Compute attention scores
        attn_weights = torch.matmul(Q, K.transpose(-2, -1))  # [B, N, N]
        attn_weights = F.softmax(attn_weights, dim=-1)       # Normalize
        attn_output = torch.matmul(attn_weights, V)          # [B, N, C]

        return attn_output


class AnyAttention(nn.Module):
    """Flexible attention module supporting cross-attention (q, k, v can be different)."""
    def __init__(self, dim, qkv_bias=False):
        super(AnyAttention, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale = dim ** (-0.5)  # Scaling factor for attention scores
        self.proj = nn.Linear(dim, dim)  # Output projection

    def get_qkv(self, q, k, v):
        """Normalize and project queries, keys, values."""
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v
    
    def forward(self, q=None, k=None, v=None):
        """
        Args:
            q: Query features, shape [B, Nq, C]
            k: Key features, shape [B, Nk, C]
            v: Value features, shape [B, Nk, C]
        
        Returns:
            attn_mask: Attention weights, shape [B, Nq, Nk]
            out: Attention output, shape [B, Nq, C]
        """
        q, k, v = self.get_qkv(q, k, v)
        
        # Compute attention scores and mask
        attn = torch.einsum("b q c, b k c -> b q k", q, k)  # [B, Nq, Nk]
        attn = F.relu(attn)  # Non-linear activation
        attn = attn * self.scale  # Scale by dimension
        attn_mask = F.softmax(attn, dim=-1)  # Normalize
        
        # Apply attention to values
        out = torch.einsum("b q k, b k c -> b q c", attn_mask, v.float())  # [B, Nq, C]
        out = self.proj(out)  # Project to output dimension

        return attn_mask, out


# ----------------------------
# MLP and Reasoning Modules
# ----------------------------
class Mlp(nn.Module):
    """Multi-layer perceptron with normalization and dropout."""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        """
        Args:
            x: Input features, shape [B, N, C]
        
        Returns:
            MLP-processed features, shape [B, N, C]
        """
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def _init_weights(self):
        """Initialize MLP weights using Xavier uniform and uniform bias."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # Initialize biases
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
        bound1 = 1 / math.sqrt(fan_in1)
        nn.init.uniform_(self.fc1.bias, -bound1, bound1)
        
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.fc2.bias, -bound2, bound2)


class SimpleReasoning(nn.Module):
    """Lightweight reasoning module for attribute grouping."""
    def __init__(self, np, ng):
        super(SimpleReasoning, self).__init__()
        self.hidden_dim = np // ng  # Hidden dimension based on group count
        self.fc1 = nn.Linear(np, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, np)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.act = nn.GELU()
        self.swish = nn.SiLU()  # Sigmoid-weighted linear unit

    def forward(self, x):
        """
        Args:
            x: Input features, shape [B, N, C]
        
        Returns:
            Reasoning-enhanced features, shape [B, N, C]
        """
        # Global pooling and MLP
        x_1 = self.fc1(self.avgpool(x).flatten(1))  # [B, hidden_dim]
        x_1 = self.act(x_1)
        x_1 = self.swish(self.fc2(x_1)).unsqueeze(-1)  # [B, np, 1]
        
        # Residual connection
        x_1 = x_1 * x + x  # [B, N, C]
        return x_1


class Tokenmix(nn.Module):
    """Token mixing module to model interactions between spatial tokens."""
    def __init__(self, np):
        super(Tokenmix, self).__init__()
        dim = 196  # Fixed token dimension
        hidden_dim = 512
        dropout_rate = 0.1
        self.norm = nn.LayerNorm(np)
        # MLP for token mixing
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features, shape [B, P, C] (P=number of tokens)
        
        Returns:
            Token-mixed features, shape [B, P, C]
        """
        residual = x
        x = self.norm(x)
        x = rearrange(x, "b p c -> b c p")  # [B, C, P]
        x = self.net(x)                     # [B, C, P]
        x = rearrange(x, "b c p -> b p c")  # [B, P, C]
        out = residual + x  # Residual connection
        return out


# ----------------------------
# Main Block and Network
# ----------------------------
class Block(nn.Module):
    """Main feature processing block with attention and reasoning."""
    def __init__(self, dim, ffn_exp=4, drop_path=0.1, num_heads=1, num_parts=0, num_g=6):
        super(Block, self).__init__()
        self.dec_attn = AnyAttention(dim, True)  # Decoder attention
        self.sattention = SelfAttention(768)     # Self-attention
        self.ffn1 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.drop_path = nn.Identity()  # Placeholder for drop path
        self.reason = Tokenmix(dim)     # Token mixing module
        self.enc_attn = AnyAttention(dim, True)  # Encoder attention
        self.group_compact = SimpleReasoning(num_parts, num_g)  # Group reasoning
        self.maxpool1d = nn.AdaptiveMaxPool1d(1)  # Global max pooling
        self.enc_ffn = Mlp(dim, hidden_features=dim, act_layer=nn.GELU)  # Encoder MLP

    def forward(self, x, parts=None):
        """
        Args:
            x: Input features, shape [B, C, P]
            parts: Attribute parts, shape [B, N_parts, C]
        
        Returns:
            feats: Processed features, shape [B, C, P]
            att_0, att_1: Attention weights for loss computation
        """
        x = rearrange(x, "b c p -> b p c")  # [B, P, C]
        
        # Cross-attention between parts and features
        attn_0, attn_out = self.enc_attn(q=parts, k=x, v=x)
        attn_0 = self.maxpool1d(attn_0).flatten(1)  # [B, ...]
        parts1 = parts + attn_out  # Residual
        parts2 = self.group_compact(parts1)
        parts_out = parts2 + self.enc_ffn(parts2) + parts1  # Enhanced parts
        
        # Second cross-attention
        parts_d = parts + parts_out
        attn_1, attn_out = self.enc_attn(q=parts_d, k=x, v=x)
        attn_1 = self.maxpool1d(attn_1).flatten(1)
        parts1_d = parts_d + attn_out
        parts_comp = self.group_compact(parts1_d)
        parts_in = parts_comp + self.enc_ffn(parts_comp) + parts1_d  # Final parts
        
        # Self-attention on parts
        parts_in = self.sattention(parts_in)
        
        # Decoder attention and feature refinement
        attn_mask, feats = self.dec_attn(q=x, k=parts_in, v=parts_in)
        feats = x + feats  # Residual
        feats = self.reason(feats)  # Token mixing
        feats = feats + self.ffn1(feats)  # MLP
        feats = rearrange(feats, "b p c -> b c p")  # [B, C, P]

        return feats, attn_0, attn_1


class BiPAZSL(nn.Module):
    """BiPAZSL Network: Integrates part attention and semantic reasoning for zero-shot learning."""
    def __init__(self, basenet, c,
                 attritube_num, cls_num, ucls_num, group_num, w2v,
                 scale=20.0, device=None): 

        super(BiPAZSL, self).__init__()
        # Dataset parameters
        self.attritube_num = attritube_num
        self.group_num = group_num
        self.feat_channel = c
        self.batch = 10
        self.cls_num = cls_num
        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num  # Number of seen classes
        
        # Semantic embeddings
        self.w2v_att = torch.from_numpy(w2v).float().to(device)  # [num_attr, w2v_dim]
        self.W = nn.Parameter(
            trunc_normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),
            requires_grad=True
        )  # Map w2v to feature dim
        self.V = nn.Parameter(
            trunc_normal_(torch.empty(self.feat_channel, self.attritube_num)),
            requires_grad=True
        )  # Map features to attribute dim
        assert self.w2v_att.shape[0] == self.attritube_num, "Mismatch in attribute count"

        # Temperature scaling
        if scale <= 0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)  # Learnable scale
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)  # Fixed scale

        # Backbone splitting (ViT)
        self.backbone_patch = nn.Sequential(*list(basenet.children()))[0]  # Patch embedding
        self.backbone_drop = nn.Sequential(*list(basenet.children()))[1]    # Dropout
        self.backbone_0 = nn.Sequential(*list(basenet.children()))[2][:-1]  # First L-1 layers
        self.backbone_1 = nn.Sequential(*list(basenet.children()))[2][-1]   # Last layer

        # Attention configuration
        self.num_heads = basenet.num_heads if hasattr(basenet, 'num_heads') else 12
        self.original_seq_len = basenet.pos_embed.shape[1]
        self.part_attention = Part_Attention(
            num_heads=self.num_heads,
            original_seq_len=self.original_seq_len
        )

        # Additional components
        self.drop_path = 0.1
        self.cls_token = basenet.cls_token  # Class token from ViT
        self.pos_embed = basenet.pos_embed  # Position embedding
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)  # Global pooling
        self.CLS_loss = nn.CrossEntropyLoss()      # Classification loss
        self.Reg_loss = nn.MSELoss()              # Regression loss

        # Main processing block
        self.blocks = Block(
            self.feat_channel,
            num_heads=1,
            num_parts=self.attritube_num,
            num_g=self.group_num,
            ffn_exp=4,
            drop_path=0.1
        )

    def _get_attn_hook(self, module, input, output):
        """Hook to collect attention weights from backbone blocks."""
        if isinstance(output, tuple) and len(output) == 2:
            self.attn_weights.append(output[1])  # Store attention weights

    def compute_score(self, gs_feat, seen_att, att_all):
        """Compute classification scores via feature-attribute similarity."""
        gs_feat = gs_feat.view(self.batch, -1)  # [B, C]
        
        # Normalize features and attributes
        gs_feat_norm = torch.norm(gs_feat, p=2, dim=1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)  # L2 normalization
        
        temp_norm = torch.norm(att_all, p=2, dim=1).unsqueeze(1).expand_as(att_all)
        seen_att_normalized = att_all.div(temp_norm + 1e-5)  # L2 normalization
        
        # Compute similarity scores
        score_o = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)  # [B, num_classes]
        score_o = score_o * self.scale  # Scale by temperature
        
        # Handle different attribute sets (seen/unseen/all)
        d, _ = seen_att.shape
        if d == self.cls_num:
            score = score_o
        elif d == self.scls_num:
            score = score_o[:, :d]
            uu = self.ucls_num
            if self.training:
                # Compute bias loss for seen/unseen separation
                mean1 = score_o[:, :d].mean(1)
                std1 = score_o[:, :d].std(1)
                mean2 = score_o[:, -uu:].mean(1)
                std2 = score_o[:, -uu:].std(1)
                mean_loss = F.relu(mean1 - mean2).mean(0) + F.relu(std1 - std2).mean(0)
                return score, mean_loss
        elif d == self.ucls_num:
            score = score_o[:, -d:]
        else:
            raise ValueError(f"Attribute count mismatch: d={d}")
        
        return score, torch.tensor(0.0, device=score.device)  # Default bias loss

    def forward(self, x, att=None, label=None, seen_att=None, att_all=None):
        """
        Forward pass with training/evaluation logic.
        Args:
            x: Input images, shape [B, 3, H, W]
            att: Attribute labels, shape [B, num_attr]
            label: Class labels, shape [B] (may include -1 for pseudo-samples)
            seen_att: Seen class attributes, shape [scls_num, num_attr]
            att_all: All class attributes, shape [cls_num, num_attr]
        
        Returns:
            score (eval mode) or loss_dict (train mode)
        """
        self.batch = x.shape[0]
        
        # Initialize attribute parts
        parts = torch.einsum('lw,wv->lv', self.w2v_att, self.W)  # [num_attr, feat_channel]
        parts = parts.expand(self.batch, -1, -1)  # [B, num_attr, feat_channel]

        # Extract patch features and add position embedding
        patches = self.backbone_patch(x)  # [B, N_patches, C]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, C]
        patches = torch.cat((cls_token, patches), dim=1)  # [B, N_patches+1, C] (include cls_token)
        feats_0 = self.backbone_drop(patches + self.pos_embed)  # [B, N, C]

        # Collect attention weights from first L-1 backbone layers
        self.attn_weights = []
        hooks = []
        for block in self.backbone_0:
            hook = block.register_forward_hook(self._get_attn_hook)
            hooks.append(hook)
        
        # Forward pass through first L-1 layers
        for block in self.backbone_0:
            feats_0, _ = block(feats_0)  # Update features
        
        # Remove hooks to avoid memory leaks
        for hook in hooks:
            hook.remove()
        
        if not self.attn_weights:
            raise ValueError("No attention weights collected. Check backbone blocks.")

        # Apply part attention to select key patches
        weighted_feats = self.part_attention(feats_0, self.attn_weights)  # [B, N, C]

        # Process with main blocks
        patches_1 = weighted_feats[:, 1:, :]  # Exclude cls_token -> [B, N_patches, C]
        feats_in = patches_1
        feats_out, att_0, att_1 = self.blocks(feats_in.transpose(1, 2), parts=parts)  # [B, C, P]

        # Final backbone layer processing
        patches_1 = torch.cat((cls_token, feats_out.transpose(1, 2)), dim=1)  # [B, N, C]
        feats_1_out = self.backbone_1(patches_1 + self.pos_embed)
        feats_1 = feats_1_out[0][:, 1:, :]  # Exclude cls_token -> [B, N_patches, C]
        feats_1, att_2, att_3 = self.blocks(feats_1.transpose(1, 2), parts=parts)  # [B, C, P]

        # Extract global features and map to attributes
        feats = feats_1
        out = self.avgpool1d(feats.view(self.batch, self.feat_channel, -1)).view(self.batch, -1)  # [B, C]
        out = torch.einsum('bc,cd->bd', out, self.V)  # [B, num_attr]

        # Compute scores and losses
        contrast_features = out
        score, bias_loss = self.compute_score(out, seen_att, att_all)

        if not self.training:
            return score  # Return scores in evaluation mode

        # Loss computation (training mode)
        label = label.to(score.device)
        valid_mask = label >= 0  # Mask for valid (non-pseudo) labels
        has_valid = valid_mask.any()

        # Regression loss (attention vs attribute labels)
        Lreg = self.Reg_loss(att_0, att) + self.Reg_loss(att_1, att) + \
               self.Reg_loss(att_2, att) + self.Reg_loss(att_3, att)
        
        # Classification loss (only for valid labels)
        if has_valid:
            Lcls = self.CLS_loss(score[valid_mask], label[valid_mask])
        else:
            Lcls = torch.tensor(0.0, device=score.device)
        
        # Semantic contrastive loss
        Lcon = semantic_contrastive_loss(contrast_features, label, self.w2v_att @ self.W)

        # Pack loss dictionary
        loss_dict = {
            'Reg_loss': Lreg,
            'Cls_loss': Lcls,
            'scale': self.scale.item(),
            'bias_loss': bias_loss,
            'con_loss': Lcon
        }

        return loss_dict


# ----------------------------
# Loss Functions
# ----------------------------
def semantic_contrastive_loss(features, labels, attr_embed, temperature=0.07):
    """
    Semantic-guided contrastive loss (handles negative labels safely).
    Args:
        features: Input features, shape [B, C]
        labels: Sample labels, shape [B] (may include -1 for pseudo-samples)
        attr_embed: Attribute embeddings, shape [num_classes, C] or [C, num_classes]
        temperature: Temperature scaling for logits
    
    Returns:
        Contrastive loss value
    """
    device = features.device
    labels = labels.to(device)

    # Return 0 if no valid labels
    valid_mask = labels >= 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    # Filter valid features and labels
    features = features[valid_mask]
    labels_valid = labels[valid_mask].clone()

    # Adjust attribute embedding shape
    if attr_embed.dim() == 2 and attr_embed.shape[0] != (torch.max(labels_valid).item() + 1):
        attr_embed = attr_embed.t()  # Transpose to [num_classes, C]
    attr_embed = attr_embed.to(device)

    # Clamp labels to avoid out-of-bounds
    labels_valid = torch.clamp(labels_valid, max=attr_embed.shape[0] - 1)

    # Normalize and compute logits
    features = F.normalize(features, dim=1)
    attr_embed = F.normalize(attr_embed, dim=1)
    logits = torch.matmul(features, attr_embed.t()) / temperature  # [B_valid, num_classes]
    loss = F.cross_entropy(logits, labels_valid)

    return loss


# ----------------------------
# Model Construction
# ----------------------------
def build_BiPAZSL(cfg):
    """Construct BiPAZSL based on configuration."""
    from model.modeling import utils
    from model.modeling.vit.vit_model import vit_base_patch16_224 as create_model

    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    
    # Dataset parameters
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    group_num = info["g"]
    c, w, h = 768, 14, 14  # Feature dimension and spatial size
    scale = cfg.MODEL.SCALE

    # Initialize ViT backbone
    vit_model = create_model(num_classes=-1)  # Disable classification head
    vit_model_path = "/root/shared-nvme/BiPAZSL/pretrain_model_vit/vit_base_patch16_224.pth"
    weights_dict = torch.load(vit_model_path)
    
    # Remove unused keys (classification head)
    del_keys = ['head.weight', 'head.bias'] if vit_model.has_logits else ['head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    vit_model.load_state_dict(weights_dict, strict=False)

    # Load attribute embeddings
    w2v_file = dataset_name + "_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)
    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return BiPAZSL(
        basenet=vit_model,
        c=c,
        scale=scale,
        attritube_num=attritube_num,
        group_num=group_num,
        w2v=w2v,
        cls_num=cls_num,
        ucls_num=ucls_num,
        device=device
    )
