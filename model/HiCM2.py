import torch
import torch.nn as nn
from .modeling_t5 import T5ForConditionalGeneration
from .vit import VisionTransformer
from transformers import T5Tokenizer
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

def _get_tokenizer(tokenizer_path, num_bins=0):
    if 't5' in tokenizer_path:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if num_bins:
            new_tokens = ["<time=" + str(i) + ">" for i in range(num_bins)]
            tokenizer.add_tokens(list(new_tokens))
    else:
        raise NotImplementedError(tokenizer_path)
    return tokenizer

class Vid2Seq(torch.nn.Module):
    def __init__(self,
                 t5_path,
                 num_features=100,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=2048,
                 vis_drop=0.,
                 tokenizer=None,
                 enc_drop=0.,
                 dec_drop=0.1,
                 use_speech=True,
                 use_video=True,
                 num_bins=100,
                 label_smoothing=0.1,
                 memory_bank=None,
                 args=None):
        super().__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(encoder_dropout=enc_drop, decoder_dropout=dec_drop, label_smoothing=label_smoothing,
                                                                   pretrained_model_name_or_path=t5_path, local_files_only=True, is_gated_act="v1_1" in t5_path,args=args)
        self.t5_model.resize_token_embeddings(len(tokenizer) - num_bins)  # remove the weights of the 28 tokens that are not used (32128 vs 32100 in the tokenizer)
        self.t5_model.resize_token_embeddings(len(tokenizer))  # add time tokens
        self.visual_encoder = VisionTransformer(num_features=num_features,
                                                embed_dim=embed_dim,
                                                depth=depth,
                                                num_heads=heads,
                                                mlp_dim=mlp_dim,
                                                qkv_bias=True,
                                                qk_scale=None,
                                                drop_rate=vis_drop,
                                                attn_drop_rate=vis_drop,
                                                norm_layer=nn.LayerNorm)
        self.t5_tokenizer = tokenizer
        self.use_speech = use_speech
        self.use_video = use_video
        self.proj_v2t = None
        if self.t5_model.model_dim != 768:
            self.proj_v2t = nn.Linear(768, self.t5_model.model_dim)

        ##################################################################
        self.memory_bank=memory_bank
        self.args=args
        
        if self.memory_bank is not None:
            if args.ret2t5_proj == "deep":
                n_input_proj=2
                txt_dim=768 #t5-large
                hidden_dim=768 #(For detr encoder size)
                input_dropout=0.5
                self.n_input_proj = n_input_proj 
                relu_args = [True] * 3
                relu_args[n_input_proj-1] = False
                self.ret2t5_proj = nn.Sequential(*[
                    LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                    LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                    LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
                ][:n_input_proj])
            elif args.ret2t5_proj == "simple":   
                self.ret2t5_proj = nn.Linear(768, self.featdim)
            

    def forward(self, video, input_tokenized, output_tokenized,mode='None',uns_video=None):

        ### ret 
        if self.memory_bank is not None:
            if isinstance(video, dict):  # cached
                target_video = video["video"]
            else:
                target_video = video
            memory_bank=self.memory_bank
            ret_texts = self.ret(target_video,memory_bank,mode,uns_video=uns_video) # B R T_D
        ### 

        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])  # B L D
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )
        
        if self.use_video:
            if isinstance(video, dict):  # cached
                video, atts_vis = video["video"], video["atts_vis"]

            else:
                video = self.visual_encoder(video)  # B T D
                if self.proj_v2t is not None:
                    video = self.proj_v2t(video)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)

            if self.args.ret_option=="hier_concat":
                video = torch.cat([video,ret_texts],dim=1)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)

            video_dict = {"video": video, "atts_vis": atts_vis}

        else:
            video_dict = None
        

        if self.use_video and self.use_speech:
            encoded.last_hidden_state = torch.cat([video, encoded.last_hidden_state], dim=1)
            encoder_atts = torch.cat([atts_vis, input_tokenized['attention_mask']], dim=1)
            
        elif self.use_video:
            encoded = BaseModelOutput(last_hidden_state=video)
            encoder_atts = atts_vis
        elif self.use_speech:
            encoder_atts = input_tokenized['attention_mask']

        targets = output_tokenized['input_ids'].masked_fill(
            output_tokenized['input_ids'] == self.t5_tokenizer.pad_token_id, -100
        )

        outputs = self.t5_model(
            encoder_outputs=encoded,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokenized['attention_mask'],
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}, video_dict

    @torch.no_grad()
    def generate(
            self,
            video,
            input_tokenized,
            use_nucleus_sampling=False,
            num_beams=4,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
            uns_video=None,
            mode=None,
    ):
        """
        Args:
            video (torch.Tensor): A tensor of shape (batch_size, T, D)
            input_tokenized (torch.Tensor): A tensor of shape (batch_size, L)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        
        ### ret 
        if self.memory_bank is not None:
            target_video = video
            memory_bank=self.memory_bank
            ret_texts = self.ret(target_video,memory_bank,mode,uns_video=uns_video) # B R T_D
        ### 
        if self.use_speech:
                    text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])  # B L D
                    encoded = self.t5_model.encoder(
                        attention_mask=input_tokenized['attention_mask'],
                        inputs_embeds=text,
                    )
        if self.use_video:
            if isinstance(video, dict):  # cached
                video, atts_vis = video["video"], video["atts_vis"]

            else:
                video = self.visual_encoder(video)  # B T D
                if self.proj_v2t is not None:
                    video = self.proj_v2t(video)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)

            if self.args.ret_option=="hier_concat":
                video = torch.cat([ret_texts,video],dim=1)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)

            video_dict = {"video": video, "atts_vis": atts_vis}

        else:
            video_dict = None
        

        if self.use_video and self.use_speech:
            encoded.last_hidden_state = torch.cat([video, encoded.last_hidden_state], dim=1)
            encoder_atts = torch.cat([atts_vis, input_tokenized['attention_mask']], dim=1)
        elif self.use_video:
            encoded = BaseModelOutput(last_hidden_state=video)
            encoder_atts = atts_vis
        elif self.use_speech:
            encoder_atts = input_tokenized['attention_mask']


        outputs = self.t5_model.generate(
                encoder_outputs=encoded,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
        )
        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return output_text
    
    def softattention_select(self,memory_bank, feature,mode,uns_video=None):
        
             
        soft_k = self.args.soft_k ## in here, soft_k means memory pool size
        if self.args.ret_option=="hier_ret" or self.args.ret_option=="hier_concat":
            if self.args.sim_match=="anchor_cos":
                window_size = self.args.window_size
                frame_length = feature.shape[1]
                segment_length = frame_length // window_size 
                topk_window_embeds = []
                total_window_sents = []
                for b in range(feature.shape[0]):  # iterate over batches
                    batch_topk_window_embeds = []
                    batch_total_window_sents = []
                    
                    for i in range(window_size):
                        start = i * segment_length
                        end = start + segment_length
                        
                        # target_feature: mean of the segment
                        target_feature = torch.mean(feature[b, start:end], dim=0)
                        
                        # Perform hierarchical memory search
                        topk_embed = self.hierarchical_memory_search(target_feature,soft_k,memory_bank)
                        batch_topk_window_embeds.append(topk_embed)
                    
                    batch_topk_window_embeds = torch.cat(batch_topk_window_embeds, dim=0).unsqueeze(0).float()  # b, window, h_dim
                    topk_window_embeds.append(batch_topk_window_embeds)
                    total_window_sents.append('no')  # Adjust this part according to your logic
                
                topk_window_embeds = torch.cat(topk_window_embeds, dim=0)  # [batch_size, window_size, h_dim]
                return topk_window_embeds, total_window_sents
        
    
    def get_topk_indices(self,similarity_scores, k):
        if similarity_scores.shape[0] < k:
            return torch.arange(similarity_scores.shape[0])
        return torch.topk(similarity_scores, k, dim=0).indices

    def hierarchical_memory_search(self, target_feature, soft_k, memory_hierarchy):
        k = soft_k
        threshold = 0.7
        selected_levels = self.args.hier_use
        retrieval_type = self.args.hier_ret_num
        
        combined_vectors = []
        topk_clusters = []
        
        max_level = max(memory_hierarchy.keys(), key=lambda x: int(x.split('_')[1]))
        sorted_levels = sorted(memory_hierarchy.keys(), key=lambda x: int(x.split('_')[1]), reverse=True)
        
        # Create a dictionary to store summaries for each level
        level_summaries = {level: [] for level in sorted_levels}
        for i,level in enumerate(sorted_levels):
            clusters = memory_hierarchy[level]
            if not topk_clusters:
                topk_clusters = [
                    (cosine_similarity(target_feature.unsqueeze(0), cluster["clip_embedding"]), cluster)
                    for cluster_id, cluster in clusters.items()
                ]
                
                topk_clusters.sort(key=lambda x: x[0], reverse=True)
                if retrieval_type == "max":
                    topk_clusters = topk_clusters[:1]
                elif retrieval_type == "top-k":
                    # topk_clusters = topk_clusters[:k]
                    topk_clusters = topk_clusters[:k]
                elif retrieval_type == "similarity":
                    topk_clusters = [(score, cluster) for score, cluster in topk_clusters if score >= threshold]
                    if not topk_clusters:
                        return torch.zeros_like(target_feature.unsqueeze(0))  # No clusters exceed the threshold

            else:
                next_level_clusters = []
                
                for _, cluster in topk_clusters:
                    parent_ids = cluster["parent_clusters"]
                    
                    if isinstance(parent_ids, list):
                        for parent_id in parent_ids:
                            if f'cluster_{parent_id}' in clusters:
                                sub_cluster = clusters[f'cluster_{parent_id}']
                                sub_score = cosine_similarity(target_feature.unsqueeze(0), sub_cluster["clip_embedding"])
                                next_level_clusters.append((sub_score, sub_cluster))
                    else:
                        if f'cluster_{parent_ids}' in clusters:
                            sub_cluster = clusters[f'cluster_{parent_ids}']
                            sub_score = cosine_similarity(target_feature.unsqueeze(0), sub_cluster["clip_embedding"])
                            next_level_clusters.append((sub_score, sub_cluster))
                
                next_level_clusters.sort(key=lambda x: x[0], reverse=True)
                if retrieval_type == "max":
                    next_level_clusters = next_level_clusters[:1]
                elif retrieval_type == "top-k":
                    next_level_clusters = next_level_clusters[:k]
                elif retrieval_type == "similarity":
                    next_level_clusters = [(score, cluster) for score, cluster in next_level_clusters if score >= threshold]
                    if not next_level_clusters:
                    # If retrieval_type is adaptive and topk_clusters is empty, return averaged combined_vectors
                        if combined_vectors:
                            final_embedding = torch.cat(combined_vectors, dim=0).mean(dim=0, keepdim=True)
                            return final_embedding
                        else:
                            return torch.zeros_like(target_feature.unsqueeze(0))  # No clusters exceed the threshold

            # Append the summary texts for the current level
            for _, cluster in topk_clusters:
                if "summary" in cluster:
                    level_summaries[level].append(cluster["summary"])
            if level in selected_levels:
                level_vectors = torch.stack([cluster["clip_embedding"] for _, cluster in topk_clusters]).squeeze(dim=1)
                if retrieval_type != "max":
                    level_vectors=level_vectors.mean(dim=0,keepdim=True)
                combined_vectors.append(level_vectors)
        # for level, summaries in level_summaries.items():
        #     print(f"Level: {level}")
        #     for summary in summaries:
        #         print(f" - Summary: {summary}")
        final_embedding = torch.cat(combined_vectors, dim=0)  
        final_embedding = final_embedding.mean(dim=0,keepdim=True)

        return final_embedding


    def ret(self,target_video,memory_bank,mode,uns_video=None):
        #######
        topk_embeds,topk_sents = self.softattention_select(memory_bank,target_video,mode,uns_video=uns_video)#topk_embeds : [batch,window,hidden]
        
        if len(topk_embeds)==0:
            return None,None,None,None

        window_size = self.args.window_size
        value_vectors=topk_embeds
        
        if len(value_vectors.shape) != 3 :
            value_vectors=torch.unsqueeze(value_vectors,dim=0)
        b=value_vectors.shape[0] #batch_size
        s=value_vectors.shape[1] #selected 
        h=value_vectors.shape[2] #hidden dimension 768
        value_vectors=value_vectors.view(b,s,h) # 1 window hid_dim
        
        if self.args.ret_encoder=="avg":
            value_vectors=topk_embeds

        ret = self.ret2t5_proj(value_vectors)  #encoder_hidden_size -> decoder_hidden_size
        return ret
        #######
        
    
class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)

class CA(torch.nn.Module):
    def __init__(self, dim=768, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.manual_seed(0)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.k.bias is not None:
            nn.init.xavier_normal_(self.k.bias)
        if self.v.bias is not None:
            nn.init.xavier_normal_(self.v.bias)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x_q, x_k, x_v):
        B, N_q, C = x_q.shape
        _, N_kv, C = x_k.shape
        _, N_kv, C = x_v.shape

        # b, h, n, d
        q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_k).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_v).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # [b, h, n, d] * [b, h, d, m] -> [b, h, n, m]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

