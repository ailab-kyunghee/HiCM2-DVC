import json
import numpy as np
import torch
import transformers
from finch import FINCH
from sklearn.metrics import normalized_mutual_info_score as nmi
import clip
import time
import pickle
import logging
from tqdm import tqdm
from transformers import logging as transformers_logging
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('your_token_here')  # Replace with your actual token
import os

def feature_save_sents(args,sentences):
    
    if len(args.target_domain)==1:
        target_domain=args.target_domain[0]
    elif len(args.target_domain)>1:
        for i in args.target_domain:
            if i == args.target_domain[0]:
                target_domain = i
            else:
                target_domain = target_domain+'_'+i
    save_path = f"./bank/{target_domain}/clip"  
    os.makedirs(save_path, exist_ok=True)
    
    # Check if the file already exists
    if os.path.exists(os.path.join(save_path, f"clip_token_embeds_{target_domain}.npy")):  
        token_embeds = np.load(os.path.join(save_path, f"clip_token_embeds_{target_domain}.npy"))  
        print("Sentence embedding number check:", token_embeds.shape[0])
        print("Sentence embedding shape:", token_embeds.shape)
        return token_embeds
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-L/14", device=device)

        token_embeds = []

        for sentence in sentences:
            try:
                token = clip.tokenize(sentence, truncate=True).to(device)
                with torch.no_grad():
                    token_embed = clip_model.encode_text(token).cpu().numpy()
                token_embeds.append(token_embed)
            except Exception as e:
                print(f"Error processing sentence: {sentence}")
                print(e)
        
        token_embeds = np.concatenate(token_embeds, axis=0)

        os.makedirs(save_path, exist_ok=True)
        
        print("Sentence embedding number check:", token_embeds.shape[0])
        print("Sentence embedding shape:", token_embeds.shape)
        np.save(os.path.join(save_path, f"clip_token_embeds_{args.target_domain}.npy"), token_embeds)
        return token_embeds

def sample_data(data, sample_size=100):
    sampled_data = {}
    keys = list(data.keys())
    sampled_keys = np.random.choice(keys, min(sample_size, len(keys)), replace=False)
    for key in sampled_keys:
        sampled_data[key] = data[key]
    return sampled_data

def finch_clustering_LLM_summarization(args):
    if len(args.target_domain) == 1:
        target_domain = args.target_domain[0]
    elif len(args.target_domain) > 1:
        for i in args.target_domain:
            if i == args.target_domain[0]:
                target_domain = i
            else:
                target_domain = target_domain + '_' + i
    print(target_domain)
    sentences = []
    if "yc2" in args.target_domain:
        with open('./yc2_train.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            for i, video in enumerate(data):
                sentences.extend(data[video]['sentences'])
    if "vitt" in args.target_domain:
        with open('./vitt_train.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            for i, video in enumerate(data):
                sentences.extend(data[video]['sentences'])
    if "anet" in args.target_domain:
        with open('./anet_train.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            for i, video in enumerate(data):
                sentences.extend(data[video]['sentences'])
    # Placeholder for feature extraction and clustering
    features = feature_save_sents(args, sentences)
    start_time_finch = time.time()

    c, num_clust, req_c = FINCH(features)

    end_time_finch = time.time()
    time_finch = end_time_finch - start_time_finch
    print(f"FINCH clustering time: {time_finch:.2f} seconds")
    exit()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize the pipeline and CLIP model
    model_id = f"meta-llama/Meta-Llama-3-{args.LLM_size}B-Instruct"
    pipeline = llama_pipeline(model_id=model_id)
    clip_model, _ = clip.load("ViT-L/14", device=device)
    hierarchical_results = {}
    cluster_indices = {}
    
    total_levels = c.shape[1]
    total_clusters = sum(len(np.unique(c[:, level])) for level in range(total_levels))
    overall_progress = tqdm(total=total_clusters, desc='Overall Progress') 

    # Level 0: Summarize all sentences
    level_0_summaries = {}
    cluster_indices[0] = {cluster_id: [] for cluster_id in np.unique(c[:, 0])}
    
    for cluster_id in np.unique(c[:, 0]):
        indices = np.where(c[:, 0] == cluster_id)[0]
        cluster_sentences = [sentences[i] for i in indices]
        if cluster_sentences:
            summary = process_chunk(pipeline, cluster_sentences)
            clip_embedding = get_clip_embedding(summary, clip_model, device)
            # Use formatted cluster_id for consistency
            level_0_summaries[f"cluster_{cluster_id}"] = {
                "indices": indices.tolist(),
                "sentences": cluster_sentences,
                "summary": summary,
                "clip_embedding": clip_embedding,
                "parent_clusters": None,  # No parent clusters for the first level
            }
            cluster_indices[0][cluster_id] = indices
        overall_progress.update(1)  

    hierarchical_results['level_1'] = level_0_summaries

    # Prepare sentences for the next level
    sentences = [level_0_summaries[cluster_id]["summary"] for cluster_id in level_0_summaries]
    
    # Process each subsequent level
    for level in range(1, c.shape[1]):
        hierarchical_results[f"level_{level + 1}"] = {}
        cluster_indices[level] = {}
        check_num_for_prev_sum = 0
        for cluster_id in np.unique(c[:, level]):
            indices = np.where(c[:, level] == cluster_id)[0]
            prev_cluster_ids = np.unique(c[indices, level - 1])
            
            # Collect sentences for the current cluster
            current_sentences = []
            seen_sentences = set()
            for prev_cluster_id in prev_cluster_ids:
                if prev_cluster_id in cluster_indices[level - 1]:
                    sentence = hierarchical_results[f"level_{level}"][f"cluster_{prev_cluster_id}"]["summary"]
                    if sentence not in seen_sentences:
                        current_sentences.append(sentence)
                        seen_sentences.add(sentence)
            # Generate summary for the current cluster
            if current_sentences:
                check_num_for_prev_sum += len(current_sentences)
                summary = process_chunk(pipeline, current_sentences)
                clip_embedding = get_clip_embedding(summary, clip_model, device)
                
                hierarchical_results[f"level_{level + 1}"][f"cluster_{cluster_id}"] = {
                    "indices": indices.tolist(),
                    "sentences": current_sentences,
                    "summary": summary,
                    "clip_embedding": clip_embedding,
                    "parent_clusters": prev_cluster_ids.tolist(),  # Track parent clusters
                }
                
                # Update cluster indices for the next level
                cluster_indices[level][cluster_id] = np.unique([i for prev_id in prev_cluster_ids 
                                                                for i in cluster_indices[level - 1][prev_id]])
            overall_progress.update(1)  
        print(f'Level {level+1}: num of prev summary sentences is {check_num_for_prev_sum} ')
        # Prepare sentences for the next level
        if level < c.shape[1] - 1:
            next_level_summaries = []
            for cluster_id in np.unique(c[:, level]):
                summary_key = f"cluster_{cluster_id}"
                if summary_key in hierarchical_results[f"level_{level + 1}"]:
                    next_level_summaries.append(hierarchical_results[f"level_{level + 1}"][summary_key]["summary"])
            if next_level_summaries:
                sentences = next_level_summaries

    overall_progress.close() 

    # Save results
    with open(f'./hierarchical_clustering_results_{target_domain}_{args.LLM_size}B.pkl', 'wb') as f:
        pickle.dump(hierarchical_results, f)
    logging.info("Clustering results saved to hierarchical_clustering_results.pkl")




def llama_pipeline(model_id=False):
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )


def prepare_prompt(sentences):
    return f"Generate Summary. Provide only the summary sentence.\n\nSentences:\n" + "\n".join(sentences)


def process_chunk(pipeline, chunk):
    prompt = prepare_prompt(chunk)
    messages = [
        {"role": "system", "content": "You are an expert in creating summary.\
        1.Generate a single sentence that best represents the given sentences.\
        2.The sentence should use a maximum of 20 words.\
        3.Provide only the information requested above. Do not include explanations, reasons for decisions, 'Note' or any additional things."},
        {"role": "user", "content": prompt},
    ]
    terminators = [
        pipeline.tokenizer.eos_token_id,
        # pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=100,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    text = outputs[0]["generated_text"][-1]['content']
    
    return text


def get_clip_embedding(text, model, device):
    text_inputs = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features.cpu().numpy()





# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Construct Feature Bank')
    parser.add_argument("--LLM_size", type=str, default=False, help="8(small) or 70(big)")
    parser.add_argument("--LLM_use",type=bool,default=False,help="If true,use LLM. If not, use average")
    parser.add_argument("--target_domain", nargs='+', default=['yc2'], help="which domain will be used in ret bank // ['vitt','yc2',...]")
    
    args = parser.parse_args()
    if len(args.target_domain)==1:
        args.target_domain=args.target_domain[0]
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    finch_clustering_LLM_summarization(args)



