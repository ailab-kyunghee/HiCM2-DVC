from .HiCM2 import Vid2Seq as HiCM2
from .HiCM2 import _get_tokenizer
from .texttitling import TextTilingTokenizer


def build_HiCM2_model(args, tokenizer,memory_bank=None):
    model = HiCM2(t5_path=args.model_name,
                    num_features=args.max_feats,
                    embed_dim=args.embedding_dim,
                    depth=args.depth,
                    heads=args.heads,
                    mlp_dim=args.mlp_dim,
                    vis_drop=args.visual_encoder_dropout,
                    enc_drop=args.text_encoder_dropout,
                    dec_drop=args.text_decoder_dropout,
                    tokenizer=tokenizer,
                    num_bins=args.num_bins,
                    label_smoothing=args.label_smoothing,
                    use_speech=args.use_speech,
                    use_video=args.use_video,
                    memory_bank=memory_bank,
                    args=args)
    return model
