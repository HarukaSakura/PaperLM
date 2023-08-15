import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from typing import Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from configuration_paperlm import PaperLMConfig
from dataclasses import dataclass
from transformers.utils import ModelOutput

logger = logging.get_logger(__name__)


PaperLMLayerNorm = nn.LayerNorm

class PathEmbeddings(nn.Module):
    """Construct the embeddings from path tags and subscripts.
    """

    def __init__(self, config):
        super(PathEmbeddings, self).__init__()
        self.max_depth = config.max_depth

        self.path_unitseq2_embeddings = nn.Linear(config.path_unit_hidden_size * self.max_depth, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.activation = nn.ReLU()
        self.path_unitseq2_inner = nn.Linear(config.path_unit_hidden_size * self.max_depth, 4 * config.hidden_size)
        self.inner2emb = nn.Linear(4 * config.hidden_size, config.hidden_size)

        self.path_tag_sub_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.max_path_tag_unit_embeddings, config.path_unit_hidden_size)
                for _ in range(self.max_depth)
            ]
        )

        self.path_subs_sub_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.max_path_subs_unit_embeddings, config.path_unit_hidden_size)
                for _ in range(self.max_depth)
            ]
        )

    def forward(self, path_tags_seq=None, path_subs_seq=None):
        path_tags_embeddings = []
        path_subs_embeddings = []

        for i in range(self.max_depth):
            path_tags_embeddings.append(self.path_tag_sub_embeddings[i](path_tags_seq[:, :, i]))
            path_subs_embeddings.append(self.path_subs_sub_embeddings[i](path_subs_seq[:, :, i]))

        path_tags_embeddings = torch.cat(path_tags_embeddings, dim=-1)
        path_subs_embeddings = torch.cat(path_subs_embeddings, dim=-1)

        path_embeddings = path_tags_embeddings + path_subs_embeddings

        path_embeddings = self.inner2emb(self.dropout(self.activation(self.path_unitseq2_inner(path_embeddings))))

        return path_embeddings

class PaperLMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(PaperLMEmbeddings, self).__init__()
        self.max_depth = config.max_depth
        self.pure_text = config.pure_text
        if not self.pure_text:
            self.path_embeddings = PathEmbeddings(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        #self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = PaperLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        

    def forward(
        self,
        input_ids=None,
        #token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        path_tags_seq=None,
        path_subs_seq=None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        #if token_type_ids is None:
            #token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        words_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)
        #token_type_embeddings = self.token_type_embeddings(token_type_ids)
    
        if self.pure_text: 
            embeddings = (
                words_embeddings
                + position_embeddings
        )
        else:
            # prepare path seq
            if path_tags_seq is None:
                path_tags_seq = self.config.tag_pad_id * torch.ones(
                    tuple(list(input_shape) + [self.max_depth]), dtype=torch.long, device=device
                )
            if path_subs_seq is None:
                path_subs_seq = self.config.subs_pad_id * torch.ones(
                    tuple(list(input_shape) + [self.max_depth]), dtype=torch.long, device=device
                )
            path_embeddings = self.path_embeddings(path_tags_seq, path_subs_seq)
            embeddings = (
                words_embeddings
                + position_embeddings
                + path_embeddings
                #+ token_type_embeddings
            )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->PaperLM
class PaperLMSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in PaperLMModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->PaperLM
class PaperLMSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->PaperLM
class PaperLMAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = PaperLMSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = PaperLMSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class PaperLMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->PaperLM
class PaperLMOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->PaperLM
class PaperLMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PaperLMAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = PaperLMAttention(config, position_embedding_type="absolute")
        self.intermediate = PaperLMIntermediate(config)
        self.output = PaperLMOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->PaperLM
class PaperLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PaperLMLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class PaperLMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->PaperLM
class PaperLMPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->PaperLM
class PaperLMLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = PaperLMPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->PaperLM
class PaperLMOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = PaperLMLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    
class PaperLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PaperLMConfig
    base_model_prefix = "PaperLM"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, PaperLMLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PaperLMEncoder):
            module.gradient_checkpointing = value

@dataclass
class PaperLMModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    mean_pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class PaperLMModel(PaperLMPreTrainedModel):
    def __init__(self, config, device="cuda:0"):
        super(PaperLMModel, self).__init__(config)
        self.embedding = PaperLMEmbeddings(config)
        self.encoder = PaperLMEncoder(config)
        self.pooler = PaperLMPooler(config)
        # Initialize weights and apply final processing
        self.post_init()
        self.device = device
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        #token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        path_tags_seq: Optional[torch.LongTensor] = None,
        path_subs_seq: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        attention_mask = attention_mask.to(self.device)
        path_tags_seq = path_tags_seq.to(self.device)
        path_subs_seq = path_subs_seq.to(self.device)
        inputs_embeds = inputs_embeds.to(self.device)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        #if token_type_ids is None:
            #token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
            
        embedding_output = self.embedding(
            input_ids=input_ids,
            position_ids=position_ids,
            #token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            path_tags_seq=path_tags_seq,
            path_subs_seq=path_subs_seq
        )  
        
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        mean_pooled_output = sequence_output.mean(dim=1)
        
        return PaperLMModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            mean_pooler_output=mean_pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        
class PaperLMModelwithProject(PaperLMPreTrainedModel):
    def __init__(self, config, encoder):
        super(PaperLMModelwithProject, self).__init__(config)
        self.encoder = encoder
        self.project_layer = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(config.hidden_size, config.hidden_size))
    def forward(self,
        attention_mask: Optional[torch.FloatTensor] = None,
        path_tags_seq: Optional[torch.LongTensor] = None,
        path_subs_seq: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, path_tags_seq=path_tags_seq, path_subs_seq=path_subs_seq).mean_pooler_output
        out = self.project_layer(out)
        return out
         
class MyCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()

    def forward(self, y, target):
        output = F.log_softmax(y,dim=-1) + 0.000000000001
        loss = F.nll_loss(output,target)

        return loss

class NRPHead(PaperLMPreTrainedModel):
    def __init__(self, config):
        super(NRPHead, self).__init__(config)
        self.num_labels =config.NRP_num_labels
        self.dense = nn.Linear(2*config.hidden_size, 2*config.hidden_size)
        self.actfn = nn.ReLU()
        self.decoder = nn.Linear(config.hidden_size, self.num_labels)
        self.bias = nn.Parameter(torch.zeros(self.num_labels))
        self.loss_func = CrossEntropyLoss()
        
    def forward(self, sequence: torch.Tensor, nr_labels):
        #attention_mask = attention_mask[:,1:-1,:]
        
        #sequence = sequence[:,1:-1,:]
        sequence_length = sequence.size(1)
        #pair_num = sequence_length*sequence_length
        pairs = []
        for i in range(sequence_length):
            for j in range(sequence_length):
                node1 = sequence[:,i,:]
                node2 = sequence[:,j,:]
                node_pair = torch.cat((node1,node2), dim=1).unsqueeze(1) # (batch_size, 1, hidden_size*2)
                pairs.append(node_pair)
        pairs = torch.cat(pairs, dim=1)# (batch_size, pair_num, hidden_size*2)
        pairs = self.dense(pairs)# (batch_size, pair_num, num_labels)
        pairs = self.actfn(pairs)
        logits = self.decoder(pairs)
        loss = self.loss_func(logits.view(-1, self.num_labels), nr_labels.view(-1))
        return logits, loss


class PQClsHead(PaperLMPreTrainedModel):
    def __init__(self, config):
        super(PQClsHead, self).__init__(config)
        self.num_labels = config.PQCls_num_labels
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.actfn = nn.ReLU()
        self.decoder = nn.Linear(config.hidden_size, self.num_labels)
        self.bias = nn.Parameter(torch.zeros(self.num_labels))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias
        self.loss_func = MyCrossEntropyLoss()
    
    def forward(self, sequence: torch.Tensor, PQClsLabel):
        sequence = self.dense(sequence)
        sequence = self.actfn(sequence)
        logits = self.decoder(sequence)
        loss = self.loss_func(logits.view(-1, self.num_labels), PQClsLabel.view(-1))
        return logits, loss


class MoCo(nn.Module):
    def __init__(self, d_rep=768, K=32768, m=0.999, T=0.07, encoder_q=None, encoder_k=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        #self.encoder_q = self.make_encoder(**encoder_params)
        #self.encoder_k = self.make_encoder(**encoder_params)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(d_rep, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.criterion = CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr.item())
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        #self.queue[:, ptr : ptr + batch_size] = keys.T
        self.queue = torch.cat([self.queue[:, :ptr], keys.T, self.queue[:, ptr + batch_size :]], dim=1).detach()
        
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, input_dict_q, input_dict_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q = self.encoder_q(**input_dict_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(**input_dict_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", *[q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", *[q, self.queue.detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        loss = self.criterion(logits, labels)

        return logits, loss


class MultiTaskModel(PaperLMPreTrainedModel):
    def __init__(self, config, device="cuda:0"):
        super(MultiTaskModel, self).__init__(config)
        self.model_q = PaperLMModel(config)
        self.model_k = PaperLMModel(config)
        self.encoder_q = PaperLMModelwithProject(config=config, encoder=self.model_q)
        self.encoder_k = PaperLMModelwithProject(config=config, encoder=self.model_k)
        self.Moco = MoCo(encoder_q=self.encoder_q, encoder_k=self.encoder_k)
        self.NRPHead = NRPHead(config)
        self.PQClsHead = PQClsHead(config)
        self.device = device
            
    def forward(
        self,
        original_encoding = None,
        augment_encoding = None,
        return_dict: Optional[bool] = None
    ):
        nrLabel = original_encoding["nodeRelation"].to(self.device)
        PQClsLabel = original_encoding["PQClsLabel"].to(self.device)
        inputs_embeds_maskpq = original_encoding["inputs_embeds_maskpq"]
        
        del original_encoding["nodeRelation"]
        del original_encoding["PQClsLabel"]
        del original_encoding["inputs_embeds_maskpq"]
        
        Moco_logits, Moco_loss = self.Moco(original_encoding, augment_encoding)

        sequence_output_for_NRP = self.model_q(**original_encoding)[0]
        original_encoding["inputs_embeds"] = inputs_embeds_maskpq
        sequence_output_for_PQCls = self.model_q(**original_encoding)[0]
        NRP_logits, NRP_loss = self.NRPHead(sequence=sequence_output_for_NRP, nr_labels=nrLabel)
        PQCls_logits, PQCls_loss = self.PQClsHead(sequence=sequence_output_for_PQCls, PQClsLabel=PQClsLabel)

        return_dict = {
            "NRP_logits": NRP_logits,
            "NRP_loss": NRP_loss,
            "PQCls_logits": PQCls_logits,
            "PQCls_loss": PQCls_loss,
            "Moco_logits": Moco_logits,
            "Moco_loss": Moco_loss
        }

        return return_dict

class Predictor(PaperLMPreTrainedModel):
    def __init__(self, config, cls_head=True, n_classes=0, output_strategy="avg", device="cuda:0"):
        super(Predictor, self).__init__(config)
        self.config = config
        self.encoder = PaperLMModel(config, device)
        self.output_strategy = output_strategy
        hidden_size = self.config.hidden_size
        self.cls_head = cls_head
        if self.cls_head:
            if n_classes == 1:
                # regression
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                    nn.Linear(hidden_size // 2, n_classes), nn.Sigmoid()
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                    nn.Linear(hidden_size // 2, n_classes))
    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        path_tags_seq: Optional[torch.LongTensor] = None,
        path_subs_seq: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        output = self.encoder(inputs_embeds=inputs_embeds, path_tags_seq=path_tags_seq, path_subs_seq=path_subs_seq, attention_mask=attention_mask)

        if self.output_strategy == "cls":
            emb = output.pooler_output
        if self.output_strategy == "avg":
            emb = output.last_hidden_state
            num_non_padding = attention_mask.sum(dim=1).unsqueeze(-1)
            non_padding_mask = attention_mask.unsqueeze(-1)
            emb = emb * non_padding_mask
            emb = emb.sum(dim=1) / num_non_padding.float()
        if self.cls_head:
            emb = self.classifier(emb)
        return emb

