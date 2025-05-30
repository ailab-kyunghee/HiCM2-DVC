# HiCM²: Hierarchical Compact Memory Modeling for Dense Video Captioning (AAAI 2025)

Official codebase for the AAAI 2025 paper:  
**"HiCM²: Hierarchical Compact Memory Modeling for Dense Video Captioning"**  
[[Paper]](https://arxiv.org/abs/2412.14585)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hicm-2-hierarchical-compact-memory-modeling/dense-video-captioning-on-youcook2)](https://paperswithcode.com/sota/dense-video-captioning-on-youcook2?p=hicm-2-hierarchical-compact-memory-modeling)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hicm-2-hierarchical-compact-memory-modeling/dense-video-captioning-on-vitt)](https://paperswithcode.com/sota/dense-video-captioning-on-vitt?p=hicm-2-hierarchical-compact-memory-modeling)

---

## 🧠 Overview

This repository provides the implementation of **HiCM²**, a memory-efficient, hierarchy-aware video-language modeling framework designed to enhance **dense video captioning** through compact temporal memory. We introduce a **hierarchical memory construction** scheme and retrieval-augmented reasoning based on temporal clustering and CLIP alignment.

---

## 📁 Directory Structure

<pre>
📁 Root
├── args.py
├── dataset/                  # Dataset loaders (YouCook2, VideoCaption, etc.)
├── dvc_eval/                # Captioning evaluation (SODA, METEOR, CIDEr, etc.)
├── model/                   # HiCM² model, backbone, and T5-related modules
├── util/                    # Utility functions (metrics, dist, t5, etc.)
├── presave/                 # Pretrained checkpoints (to be downloaded)
│   ├── vid2seq_htmchapters.pth
│   ├── vid2seq_htm.pth
│   ├── vitt/best_model.pth
│   └── yc2/best_model.pth
├── data/                    # YouCook2, ViTT Data (to be downloaded)
│   ├── vitt/*
│   └── yc2/*
├── finch-llama_hier.py      # Hierarchical memory constructor (our main contribution)
├── finch-llama_hier.sh      # Shell script to run memory constructor
├── train_ret_yc2_hier.sh    # Training script for YC2 with HiCM²
├── train_ret_vitt_hier.sh   # Training script for VITT with HiCM²
├── eval_ret_yc2_hier.sh     # Evaluation script for YC2 with HiCM²
├── eval_ret_vitt_hier.sh    # Evaluation script for VITT with HiCM²
├── hierarchical_clustering_results_yc2_70B.pkl   # Hierarchical Memory for YC2
├── hierarchical_clustering_results_vitt_70B.pkl  # Hierarchical Memory for VITT
├── requirements.txt
└── README.md
</pre>

---

## 🛠️ Setup

We recommend using a Conda environment:

```bash
conda create --name HiCM2 python=3.7
conda activate HiCM2
pip install -r requirements.txt
```

---

## 📂 Dataset Download

| Dataset | Download Link                             | Save Path     |
|---------|--------------------------------------------|----------------|
| YC2     | [Hugging Face](https://huggingface.co/datasets/Geppa/HiCM2/tree/main)   | `data/yc2/*`    |
| VITT    | [Hugging Face](https://huggingface.co/datasets/Geppa/HiCM2/tree/main) | `data/vitt/*`   |

> 📌 *Please download the corresponding dataset files and place them in the above directories to match the training/evaluation scripts.*

---

## 🔗 Pretrained Models

Due to large file sizes, pretrained weights are provided via external download:

| Model Type         | Dataset      | Download Link                                  | Save Path                               |
|--------------------|--------------|------------------------------------------------|------------------------------------------|
| Ours               | YC2          | [Hugging Face](https://huggingface.co/Geppa/HiCM2/tree/main)  | `presave/yc2/best_model.pth`             |
| Ours               | VITT         | [Hugging Face](https://huggingface.co/Geppa/HiCM2/tree/main) | `presave/vitt/best_model.pth`            |
| Vid2Seq Baseline   | HTM-Chapters | [Hugging Face](https://huggingface.co/Geppa/HiCM2/tree/main)   | `presave/vid2seq_htmchapters.pth`        |


Make sure to place downloaded `.pth` files in the correct subdirectories as shown above.

---

## 🧩 Hierarchical Memory Construction

We provide a script to construct **Hierarchical Compact Memory** with clustering and representation selection:

```bash
bash finch-llama_hier.sh
```

This will generate clustering outputs like:

- `hierarchical_clustering_results_yc2_8B.pkl`
- `hierarchical_clustering_results_vitt_70B.pkl`

You can modify the backbone, dataset, or levels within `finch-llama_hier.py`.

Note: This code is based on Hugging Face's LLaMA 3 model. You must first obtain access to LLaMA 3 from Hugging Face, and then insert your Hugging Face token into finch-llama_hier.py.

---

## 🚀 Training & Evaluation

Example (YouCook2 + Hierarchical Memory):

```bash
bash train_ret_yc2_hier.sh
```

Evaluate on YouCook2:

```bash
bash eval_ret_yc2_hier.sh
```

---

## 📊 Evaluation Metrics

We support standard dense video captioning metrics:

- **SODA**
- **METEOR**
- **CIDEr**
- **ROUGE-L**
- **BLEU-4**

Evaluation is handled via the scripts in `dvc_eval/`.

---

## 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{kim2025hicm2,
  title={HiCM$^2$: Hierarchical Compact Memory Modeling for Dense Video Captioning},
  author={Kim, Minkuk and Kim, Hyeon Bae and Moon, Jinyoung and Choi, Jinwoo and Kim, Seong Tae},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={4},
  pages={4293--4301},
  year={2025}
}
```

---

## 📝 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙏 Acknowledgements

Our framework builds upon prior work from [Vid2Seq](https://github.com/antoyang/VidChapters), [CLIP](https://github.com/openai/CLIP), and others. We thank the open-source community!
