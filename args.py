import argparse
import os

DATA_DIR = "/local_datasets/video_data"


PRESAVE_DIR = "./presave"
MODEL_DIR = "./checkpoints"

name2folder = {
    "youcook": "yc2",
    "htm": "howto100m",
    "chapters": "AllChapters",
    "vitt": "vitt",
    "anet":"anet"
}

#we do not use SSD_DIR and NLTK_FOLDER path in this code.
SSD_DIR = "/path/to/ssd_dir"
NLTK_FOLDER = "TOFILL"
def get_args_parser():
    parser = argparse.ArgumentParser("Set Vid2Seq", add_help=False)

    parser.add_argument("--cluster", type=str ,default=False,help='cluster or ...')
    parser.add_argument("--ret_path", type=str ,default=False,help='memory path, if True, load directly.')
    parser.add_argument("--hier_level",type=int,default=0)
    parser.add_argument("--LLM_ver",type=int,default=8)
    parser.add_argument("--hier_ret_num", type=str ,default='top-k',help='max,top-k,adaptive')
    parser.add_argument(
        "--hier_use",
        nargs="+",
        help="1~n",
        required=True,
    )

    parser.add_argument("--ret_encoder", type=str , default='avg', help="avg,miniTE,top1,attention")
    parser.add_argument("--sampling", type=str , default='origin', help="origin,average,max")
    parser.add_argument("--ret_option", type=str , default='hier_concat', help="hier_concat,no_ret")
    parser.add_argument("--sim_match", type=str , default='anchor_cos', help="anchor_cos,attn,multi_attn,...")
    parser.add_argument("--soft_k", type=int,default=10)
    parser.add_argument("--ret2t5_proj", type=str, default='deep',help="simple,deep")
    parser.add_argument("--bank_path",type = str, default = "/local_datasets/caption/bank")
    parser.add_argument("--bank_type", nargs='+', default=['anet'], help="which domain will be used in ret bank // ['anet','yc2','vitt']")
    parser.add_argument("--window_size", type=int, default=10,help="window number")
    parser.add_argument("--drop_last_enable",type=bool,default=False,help="if True, drop last of DataLoader will be activated.(for batch processing)")

    # Dataset specific
    parser.add_argument(
        "--combine_datasets",
        nargs="+",
        help="list of datasets to combine for training",
        required=True,
    )
    parser.add_argument(
        "--combine_datasets_val",
        nargs="+",
        help="list of datasets to combine for eval",
        default=[],
    )

    parser.add_argument(
        "--howto100m_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["htm"], "htm_vid2seq.csv"),
    )
    parser.add_argument(
        "--howto100m_features_path",
        default=os.path.join(SSD_DIR, "howto100m_clip_features"),
    )
    parser.add_argument(
        "--howto100m_subtitles_path",
        default=os.path.join(SSD_DIR, "htm_sentences"),
    )


    parser.add_argument(
        "--youcook_features_path",
        default=os.path.join(DATA_DIR, name2folder["youcook"], "vid2seq/clipvitl14.pth"),
    )
    parser.add_argument(
        "--youcook_train_json_path",
        default=os.path.join(DATA_DIR, name2folder["youcook"], "vid2seq/train.json"),
    )
    parser.add_argument(
        "--youcook_val_json_path",
        default=os.path.join(DATA_DIR, name2folder["youcook"], "vid2seq/val.json"),
    )
    parser.add_argument(
        "--youcook_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["youcook"], "vid2seq/youcook2_asr_align_proc.pkl"),
    )
    
    parser.add_argument(
        "--anet_features_path",
        default=os.path.join(DATA_DIR, name2folder["anet"], "vid2seq/clipvitl14.pth"),
    )
    parser.add_argument(
        "--anet_train_json_path",
        default=os.path.join(DATA_DIR, name2folder["anet"], "vid2seq/train.json"),
    )
    parser.add_argument(
        "--anet_val_1_json_path",
        default=os.path.join(DATA_DIR, name2folder["anet"], "vid2seq/val_1.json"),
    )
    parser.add_argument(
        "--anet_val_2_json_path",
        default=os.path.join(DATA_DIR, name2folder["anet"], "vid2seq/val_2.json"),
    )
    parser.add_argument(
        "--anet_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["anet"], "vid2seq/youcook2_asr_align_proc.pkl"),
    )
    
    parser.add_argument(
        "--vitt_features_path",
        default=os.path.join(DATA_DIR, name2folder["vitt"], "vid2seq/clipvitl14.pth"),
    )
    parser.add_argument(
        "--vitt_train_json_path",
        default=os.path.join(DATA_DIR, name2folder["vitt"], "vid2seq/train.json"),
    )
    parser.add_argument(
        "--vitt_val_json_path",
        default=os.path.join(DATA_DIR, name2folder["vitt"], "vid2seq/dev.json"),
    )
    parser.add_argument(
        "--vitt_test_json_path",
        default=os.path.join(DATA_DIR, name2folder["vitt"], "vid2seq/test.json"),
    )
    parser.add_argument(
        "--vitt_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["vitt"], "vid2seq/subtitles_align_proc.pkl"),
    )

    parser.add_argument(
        "--chapters_features_path",
        default=os.path.join(SSD_DIR, "chapters_clipvitl14_features"),
    )
    parser.add_argument(
        "--chapters_train_json_path",
        default=os.path.join(DATA_DIR, name2folder["chapters"], "chapters_dvc_train.json"),
    )
    parser.add_argument(
        "--chapters_val_json_path",
        default=os.path.join(DATA_DIR, name2folder["chapters"], "chapters_dvc_val.json"),
    )
    parser.add_argument(
        "--chapters_test_json_path",
        default=os.path.join(DATA_DIR, name2folder["chapters"], "chapters_dvc_test.json"),
    )
    parser.add_argument(
        "--chapters_subtitles_path",
        default=os.path.join(SSD_DIR, "allchapters_asr"),
    )

    # Training hyper-parameters
    parser.add_argument(
        "--denoising", default=1., type=float, help="denoising loss coef"
    )
    parser.add_argument(
        "--generative", default=1., type=float, help="generative loss coef"
    )
    parser.add_argument("--genasr", action="store_true", help="baseline that generates asr and not chapters")
    parser.add_argument("--random", action="store_true", help="random baseline")
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.25,
        help="masking probability for the denoising objective",
    )
    parser.add_argument(
        "--mask_len",
        type=int,
        default=5,
        help="masking average span length for the denoising objective",
    )
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument(
        "--beta1", default=0.9, type=float, help="Adam optimizer parameter"
    )
    parser.add_argument(
        "--beta2", default=0.999, type=float, help="Adam optimizer parameter"
    )
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size used for training"
    )
    parser.add_argument(
        "--batch_size_val",
        default=2,
        type=int,
        help="batch size used for eval",
    )
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument(
        "--epochs", default=20, type=int, help="number of training epochs"
    )
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument(
        "--label_smoothing", default=0.1, type=float, help="label smoothing"
    )
    parser.add_argument(
        "--clip_max_norm", default=1., type=float, help="gradient clipping max norm"
    )
    parser.add_argument(
        "--schedule",
        default="",
        choices=["", "cosine_with_warmup"],
        help="learning rate decay schedule, default is constant",
    )
    parser.add_argument(
        "--fraction_warmup_steps",
        default=0.1,
        type=float,
        help="fraction of number of steps used for warmup when using cosine schedule",
    )
    parser.add_argument(
        "--eval_skip",
        default=3,
        type=int,
        help='do evaluation every "eval_skip" epochs',
    )
    parser.add_argument(
        "--eval_skip2",
        default=5,
        type=int,
        help='do evaluation every "eval_skip" epochs',
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=100,
        help="print log every print_freq iterations",
    )

    # Run specific
    parser.add_argument(
        "--save_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--presave_dir",
        default=PRESAVE_DIR,
        help="the actual save_dir is an union of presave_dir and save_dir",
    )
    parser.add_argument("--device", default="cuda", help="device to use")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--load",
        default="",
        help="path to load checkpoint",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue training if loading checkpoint",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="only run evaluation")
    parser.add_argument(
        "--num_workers", default=3, type=int, help="number of workers for dataloader"
    )

    # Distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        default="t5-base",
        choices=(
            "t5-base", os.path.join(MODEL_DIR, "7BHF"), "Salesforce/blip2-flan-t5-xl"
        ),
    )
    parser.add_argument(
        "--bert_name",
        default="bert-base-uncased",
        choices=(
            "bert-base-uncased"
        ),
    )
    parser.add_argument(
        "--text_encoder_dropout", default=0.1, type=float, help="dropout to use in the text encoder"
    )
    parser.add_argument(
        "--text_decoder_dropout", default=0.1, type=float, help="dropout to use in the text decoder"
    )
    parser.add_argument(
        "--visual_encoder_dropout", default=0.1, type=float, help="dropout to use in the visual encoder"
    )
    parser.add_argument(
        "--max_feats",
        type=int,
        default=100,
        help="maximum number of video features considered, one per frame",
    )
    parser.add_argument(
        "--features_dim",
        type=int,
        default=768,
        help="dimension of the visual embedding space",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="dimension of the language modeling space",
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=2048,
        help="dimension of the visual encoder mlp",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="number of layers of visual encoder",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=12,
        help="number of heads of visual encoder",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=100,
        help="number of quantization bins for the time tokens",
    )
    parser.add_argument(
        "--no_video",
        dest="use_video",
        action="store_false",
        help="disables usage of video",
    )
    parser.add_argument(
        "--no_speech",
        dest="use_speech",
        action="store_false",
        help="disables usage of speech",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=1000,
        help="maximum number of tokens in the input speech",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=256,
        help="maximum number of tokens in the output sequence of dense captions",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="beam search size",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.,
        help="length penalty for beam search",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.,
        help="repetition penalty for beam search",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="nucleus sampling parameter",
    )
    # BLIP-2 Model parameters
    parser.add_argument(
        "--blip2_model_name",
        default="pretrain_flant5xl_vitL",
        choices=(
            "pretrain_flant5xl_vitL"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="spatial resolution for raw video",
    )
    parser.add_argument(
        "--video_example",
        default="",
        type=str,
        help="path to a video example for demo",
    )
    parser.add_argument(
        "--asr_example",
        default="",
        type=str,
        help="path to a ASR example for demo",
    )

    return parser


