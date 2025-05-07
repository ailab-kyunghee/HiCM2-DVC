import os
import numpy as np
import pickle




def load_hier_clip_memory_bank(args):
    print("######## Total Bank Type : ",args.bank_type)

    if "yc2" in args.bank_type:
        if args.ret_path:
            with open(args.ret_path, 'rb') as file:
                hier_bank=pickle.load(file)
        else:
            with open('./hierarchical_clustering_results_yc2_{}B.pkl'.format(args.LLM_ver), 'rb') as file:
                hier_bank=pickle.load(file)
    elif "vitt" in args.bank_type:
        if args.ret_path:
            with open(args.ret_path, 'rb') as file:
                hier_bank=pickle.load(file)
        else:
            with open('./hierarchical_clustering_results_vitt_{}B.pkl'.format(args.LLM_ver), 'rb') as file:
                hier_bank=pickle.load(file)
    elif "anet" in args.bank_type:
        if args.ret_path:
            with open(args.ret_path, 'rb') as file:
                hier_bank=pickle.load(file)
        else:
            with open('./hierarchical_clustering_results_anet_{}B.pkl'.format(args.LLM_ver), 'rb') as file:
                hier_bank=pickle.load(file)            
    return hier_bank