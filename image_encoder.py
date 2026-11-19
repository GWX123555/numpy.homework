"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
         
        self.qkv_weight = weights[prefix + '.in_proj_weight'].astype(np.float32)
        self.qkv_bias = weights[prefix + '.in_proj_bias'].astype(np.float32)
        self.out_weight = weights[prefix + '.out_proj.weight'].astype(np.float32)
        self.out_bias = weights[prefix + '.out_proj.bias'].astype(np.float32)
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02."""
        B, N, D = x.shape
        # 拆分维度：[B, N, H, D/H]
        x = x.reshape(B, N, self.heads, self.head_dim)
        # 调换维度：[B, H, N, D/H]
        return x.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a
         
        scale = np.float32(np.sqrt(self.head_dim))
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)

        # TODO-03b
        context = coefficients @ v
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.width)

        # TODO-03c
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'].astype(np.float32), weights[prefix+'.ln_1.bias'].astype(np.float32))
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'].astype(np.float32), weights[prefix+'.ln_2.bias'].astype(np.float32))
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        
        
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight'].astype(np.float32)
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias'].astype(np.float32)
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight'].astype(np.float32)
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias'].astype(np.float32)
        
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        update = self.attention(self.norm1(x), causal=causal)
        # TODO-04a
        residual = x + update
        u = self.norm2(residual)
        # TODO-04b
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
         
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072).astype(np.float32)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768).astype(np.float32)
        self.position = weights['visual.positional_embedding'].astype(np.float32)
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'].astype(np.float32), weights['visual.ln_pre.bias'].astype(np.float32))
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'].astype(np.float32), weights['visual.ln_post.bias'].astype(np.float32))
        self.projection = weights['visual.proj'].astype(np.float32)

    def patchify(self, images):
        """TODO-01."""
        images = images.astype(np.float32)
        B = images.shape[0]
        # 1. 拆解: [B, C, rows, p_h, cols, p_w] = [B, 3, 7, 32, 7, 32]
        x = images.reshape(B, 3, 7, 32, 7, 32)
        # 2. 换轴: 必须把行(2)和列(4)移到C(1)的前面，然后保留后面像素
        # 目标形状：[B, rows, cols, C, p_h, p_w] -> 对应轴 (0, 2, 4, 1, 3, 5)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        # 3. 展平
        return x.reshape(B, 49, 3 * 32 * 32)

    def embed(self, images):
        patches = self.patchify(images)
        # TODO-05a
        projected = patches @ self.patch_weight.T
        cls = np.tile(self.cls, (images.shape[0], 1, 1))
        # TODO-05b
        tokens = np.concatenate([cls, projected], axis=1)
        return tokens + self.position

    def encode_tokens(self, images):
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        tokens = self.encode_tokens(images)
        # TODO-06
        cls_token = tokens[:, 0, :]
        cls_normed = self.post_norm(cls_token)
        output = cls_normed @ self.projection
         
        return unit_vector(output).astype(np.float32)