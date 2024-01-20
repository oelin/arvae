from typing import Optional, Tuple

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


# TODO: Add ALiBi to attention modules.


class Attention(nn.Module):
    """Attention.

    Example
    -------
    >>> module = Attention(
    ...     embedding_dimension=256,
    ...     heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of attention heads.
        """

        super().__init__()

        self.heads = heads

        self.linears = nn.ModuleList([
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * 3,
                bias=False,
            ),
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension,
                bias=False,
            ),
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        mask : torch.Tensor
            The attention mask.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.linears[0](x)
        x = rearrange(x, 'b t (n h e) -> n b h t e', h=self.heads, n=3)
        x = F.scaled_dot_product_attention(*x, mask)
        x = self.linears[1](rearrange(x, 'b h t e -> b t (h e)'))

        return x
    

class CrossAttention(nn.Module):
    """Cross-attention.

    Example
    -------
    >>> module = CrossAttention(
    ...     embedding_dimension=256,
    ...     heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> y = torch.randn((1, 20, 256))
    >>> x = module(x, y)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of attention heads.
        """

        super().__init__()

        self.heads = heads

        self.linears = nn.ModuleList([
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension,
            ) for _ in range(4)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The target input tensor.
        y : torch.Tensor
            The source input tensor.
        mask : torch.Tensor
            The attention mask.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        q = rearrange(self.linears[0](x), 'b t (h e) -> b h t e', h=self.heads)
        k = rearrange(self.linears[1](x), 'b s (h e) -> b h s e', h=self.heads)
        v = rearrange(self.linears[2](x), 'b s (h e) -> b h s e', h=self.heads)

        x = F.scaled_dot_product_attention(q, k, v, mask)
        x = self.linears[3](rearrange(x, 'b h t e -> b t (h e)'))

        return x


class MLP(nn.Module):
    """MLP.

    Example
    -------
    >>> module = MLP(embedding_dimension=256)
    >>> x = torch.tensor((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * 3,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=embedding_dimension * 3,
                out_features=embedding_dimension,
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        return self.layers(x)


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block.

    Example
    -------
    >>> module = TransformerEncoderBlock(
    ...     embedding_dimension=256,
    ...     heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of attention heads.
        """

        super().__init__()

        self.attention = Attention(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

        self.mlp = MLP(embedding_dimension=embedding_dimension)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(normalized_shape=embedding_dimension),
            nn.LayerNorm(normalized_shape=embedding_dimension),
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        mask : torch.Tensor
            The attention mask.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = x + self.attention(self.layer_norms[0](x), mask=mask)
        x = x + self.mlp(self.layer_norms[1](x))

        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block.

    Example
    -------
    >>> module = TransformerDecoderBlock(
    ...     embedding_dimension=256,
    ...     heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> y = torch.randn((1, 20, 256))
    >>> x = module(x, y)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of attention heads.
        """

        super().__init__()

        self.attention = Attention(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

        self.cross_attention = CrossAttention(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

        self.mlp = MLP(embedding_dimension=embedding_dimension)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(normalized_shape=embedding_dimension),
            nn.LayerNorm(normalized_shape=embedding_dimension),
            nn.LayerNorm(normalized_shape=embedding_dimension),
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The target input tensor.
        y : torch.Tensor
            The source input tensor.
        mask : torch.Tensor
            The attention mask.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = x + self.attention(self.layer_norms[0](x), mask=mask)
        x = x + self.cross_attention(self.layer_norms[1](x), y, mask=None)
        x = x + self.mlp(self.layer_norms[2](x))

        return x


class TransformerEncoderDecoder(nn.Module):
    """Transformer kernel.

    Example
    -------
    >>> module = TransformerEncoderDecoder(
    ...     embedding_dimension=256,
    ...     heads=16,
    ...     layers=16,
    ... )
    >>> xs = torch.randn((1, 10, 256))
    >>> xt = torch.randn((1, 20, 256))
    >>> xt = module(xs, xt, mask=None)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        heads: int,
        layers: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of attention heads.
        layers : int
            The number encoder/decoder layers.
        """

        super().__init__()

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dimension=embedding_dimension,
                heads=heads,
            ) for _ in range(layers)
        ])

        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(
                embedding_dimension=embedding_dimension,
                heads=heads,
            ) for _ in range(layers)
        ])

    def forward(
        self, 
        xs: torch.Tensor, 
        xt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        xs : torch.Tensor
            The source input tensor.
        xt : torch.Tensor
            The target input tensor.
        mask : torch.Tensor
            The attention mask.
        
        Returns
        -------
        xt : torch.Tensor
            The target output tensor.
        """

        x = xs

        for encoder_block in self.encoder:
            x = encoder_block(x, mask=None)
        
        y = x
        x = xt
        
        for decoder_block in self.decoder:
            x = decoder_block(x, y, mask=mask)
        
        return x


@dataclass(frozen=True)
class TransformerConfiguration:
    embedding_dimension: int
    heads: int
    layers: int
    vocabulary_size: int


class Transformer(nn.Module):
    """Transformer.

    Example
    -------
    >>> configuration = TransformerConfiguration(
    ...     embedding_dimension=256,
    ...     heads=16,
    ...     layers=16,
    ...     vocabulary_size=1024,
    ... )
    >>> module = Transformer(configuration=configuration)
    >>> source = torch.tensor([1, 2, 3, 4]) 
    >>> target = torch.tensor([5, 6, 7])
    >>> logits = module(source, target, mask=None)
    """

    def __init__(self, *, configuration: TransformerConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : TransformerConfiguration
            The module configuration.
        """

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=configuration.vocabulary_size,
            embedding_dim=configuration.embedding_dimension,
        )

        self.encoder_decoder = TransformerEncoderDecoder(
            embedding_dimension=configuration.embedding_dimension,
            heads=configuration.heads,
            layers=configuration.layers,
        )
        
    def forward(
        self, 
        source: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Forward the module.

        Parameters
        ----------
        source : torch.Tensor
            The input source tokens.
        target : torch.Tensor
            The input target tokens.
        mask : torch.Tensor
            The attention mask.
        
        Returns
        -------
        logits : torch.Tensor
            The logits for each token.
        """

        xs = self.embedding(source)
        xt = self.embedding(target)
        x = self.encoder_decoder(xs, xt, mask)
        logits = F.log_softmax(x @ self.embedding.weight.T, dim=-1)

        return logits


@dataclass(frozen=True)
class VSeq2SeqConfiguration:
    embedding_dimension: int
    heads: int
    layers: int
    encoder_vocabulary_size: int
    decoder_vocabulary_size: int


class VSeq2Seq(nn.Module):
    """Variational seq2seq.

    Example
    -------
    >>> configuration = VSeq2SeqConfiguration(
    ...     embedding_dimension=256,
    ...     heads=16,
    ...     layers=16,
    ...     encoder_vocabulary_size=256,
    ...     decoder_vocabulary_size=1024,
    ... )
    >>> module = VSeq2Seq(configuration=configuration)
    >>> source = torch.tensor([1, 2, 3, 4])
    >>> logits, latent_logits, latent = module(source)
    """

    # TODO
