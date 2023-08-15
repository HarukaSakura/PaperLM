
from transformers import PretrainedConfig

class PaperLMConfig(PretrainedConfig):

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=64, 
        max_length=64,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        max_2d_position_embeddings=1024,
        path_unit_hidden_size=32,
        max_path_subs_unit_embeddings=1024, 
        max_path_tag_unit_embeddings=256,
        max_depth=10,
        max_width=100,
        subs_pad_id=101,
        tag_pad_id=5,
        NRP_num_labels=7,
        PQCls_num_labels=3,
        moco_dim=128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_length = max_length
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.path_unit_hidden_size=path_unit_hidden_size
        self.max_path_subs_unit_embeddings=max_path_subs_unit_embeddings
        self.max_path_tag_unit_embeddings=max_path_tag_unit_embeddings
        self.max_depth=max_depth
        self.max_width = max_width
        self.subs_pad_id=subs_pad_id
        self.tag_pad_id=tag_pad_id
        self.NRP_num_labels=NRP_num_labels
        self.PQCls_num_labels=PQCls_num_labels
        self.moco_dim=moco_dim