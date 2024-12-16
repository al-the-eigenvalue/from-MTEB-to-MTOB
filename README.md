# From MTEB to MTOB: Retrieval-Augmented Classification for Descriptive Grammars
This is the repository for the paper "From MTEB to MTOB: Retrieval-Augmented Classification for Descriptive Grammars".

![image](https://github.com/user-attachments/assets/f3e322c2-0a6e-4694-8e47-ed8bebe26624)

The paper contains two benchmarks:

- **The benchmark for rerankers** 

   14 grammars
   
   It is available in the ```rerankers_benchmark``` folder in this repository. Each file has top 50 paragraphs from a the grammar, ranked by BM25 using the summary for the English Wikipedia article “Word order” (as of September 3rd, 2024) as the query. Each paragraph is annotated according to its relevance to the word order of subject, object, and verb.

- **The benchmark for RAG**

  148 grammars for Word Order, Standard Negation, Polar Questions 
  \+ 148 grammars for Number of Cases
  
  It cannot be published as open-source, as stated in the Ethical Considerations section of our paper. However, we publish the metadata for this benchmark, including titles of the grammars, the authors' names, the annotations for each feature, and the pages where relevant information for each feature can be found:
  [```ground_truth_rag.csv```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/ground_truth_rag.csv)
  [```ground_truth_rag_number_of_cases.csv```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/ground_truth_rag_number_of_cases.csv)

All scripts apart from one are designed to be executed in Google Colab for convenience.

Steps for reproduction of the experiments:

### 1. Save the benchmark for RAG on Google Drive

In order to execute the Colab scripts, place the folders with the benchmark for the RAG pipeline on Google Drive, resulting in the following directories:
```
'/content/drive/MyDrive/Grammars Benchmark'
'/content/drive/MyDrive/Grammars Benchmark: Number of Cases'
```
The benchmark for RAG cannot be published as open-source. 

Since each ARR submission can be accompanied by one archive containing data (max. 200MB), and the size of the archive with the total dataset is >2GB, we provide a small sample of grammars in the archive to the anonymous reviewers.

### 2. Extract paragraphs from the grammars
Run the script [```extract_paragraphs.ipynb```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/extract_paragraphs.ipynb) in Google Colab in order to create the new folder with the paragraphs extracted from the grammars:
```
'/content/drive/MyDrive/Grammars Paragraphs/'
```
Extraction of paragraphs using BM25 reqires tokenization and removal of stop words. The model used for tokenizing paragraphs from grammars is ```en_core_web_sm``` from the [spaCy](https://github.com/explosion/spaCy) library, and the list of stop words is ```stop_words``` from ```spacy.lang.en```. Regarding BM25 itself, we use ```BM25Okapi``` from [rank_bm25](https://github.com/dorianbrown/rank_bm25) with default parameters.

Since the results from Step 1 are not reproducible with a limited sample of grammars, we provide the ```Grammars Paragraphs``` folder in the archive for the reviewers.

### 3. Install dependencies for the benchmark for rerankers

The script [```benchmarking_rerankers.ipynb```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/benchmarking_rerankers.ipynb) cannot be run on the Free Tier in Google Colab, since it requires a GPU with 40GB of VRAM.

Installing requirements:

```
cd from-MTEB-to-MTOB
pip install -r requirements.txt
```

Requirements for [BAAI/bge-en-icl](https://huggingface.co/BAAI/bge-en-icl):

```
cd ..
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install -e .
```

Pre-loading the rerankers from HuggingFace:

```
pip install -U "huggingface_hub[cli]"
huggingface-cli download BAAI/bge-en-icl
huggingface-cli download dunzhang/stella_en_1.5B_v5
huggingface-cli download nvidia/NV-Retriever-v1
huggingface-cli download Alibaba-NLP/gte-Qwen2-7B-instruct
huggingface-cli download Linq-AI-Research/Linq-Embed-Mistral
huggingface-cli download Salesforce/SFR-Embedding-2_R
huggingface-cli download Salesforce/SFR-Embedding-Mistral
```

### 4. Choose the best reranker and run it on the paragraphs

This step corresponds to Section 4 of our paper: The Benchmark for Rerankers.

  

Download the folder ```'Grammars Paragraphs'``` from Google Drive and place it into the directory where the script [```benchmarking_rerankers.ipynb```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/benchmarking_rerankers.ipynb) is to be run.

  

Run [```benchmarking_rerankers.ipynb```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/benchmarking_rerankers.ipynb), which

- chooses the best reranker by running them all on the benchmark for rerankers: approximately 1.8 GPU hours on 1x NVIDIA A100 40GB;

- runs the best reranker on the benchmark for RAG (the result from step 2, ```'Grammars Paragraphs'```): approximately 7.2 GPU hours on 1x NVIDIA A100 40GB.

  

⚠️ As of Dec 13th, 2024, inference for [bge-en-icl](https://huggingface.co/BAAI/bge-en-icl) is slow (evaluating the model on the benchmark for rerankers takes ~13 hours instead of 15 minutes) and produces results inconsistent with those obtained in September 2024. The metrics for [bge-en-icl](https://huggingface.co/BAAI/bge-en-icl) in the ```rerankers_metrics``` folder in the repository are from September 2024.

  

Upload the resulting folder ```'Reranker Similarity Scores'``` to Google Drive:

```
'/content/drive/MyDrive/Reranker Similarity Scores/'
```
### 5. Rerank the paragraphs according to scores

All scripts in Steps 5, 6, 7 can be executed in Colab.

Run [```rerank_paragraphs.ipynb```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/rerank_paragraphs.ipynb) to obtain the paragraphs reranked according to the scores obtained in Step 4. The reranked paragraphs will appear in the ```'Grammars Paragraphs/[Feature]/Reranker 20'``` subfolder for each of the four features.

### 6. Run the RAG experiments

This step corresponds to Section 5 of our paper: The Benchmark for RAG.

In order to run the RAG experiments on GPT-4o, setup an OpenAI API key in Colab Secrets and run [```rag_inference.ipynb```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/rag_inference.ipynb).

⚠️ Running ```gpt-4o-2024-05-13``` on all RAG configurations requires approximately $600 in total (including the optional Step 7: Ablation).

We provide our results of running this script in the ```RAG Results``` folder in this repository.

### 6. Calculate the metrics for RAG

In order to calculate F1 scores for all RAG configurations (Table 4 in the paper) and visualize confusion matrices, run the [```rag_metrics.ipynb```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/rag_metrics.ipynb) script.

### 7. Optional: Ablation

In order to reproduce the ablation experiment (Appendix I in the paper), run the scripts
- [```extract_paragraphs_ablation.ipynb```](https://github.com/al-the-eigenvalue/from-MTEB-to-MTOB/blob/main/extract_paragraphs_ablation.ipynb): extracting the ground-truth pages containing the information for each feature;
- [```rag_inference_ablation.ipynb```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/rag_inference_ablation.ipynb): running RAG with GPT-4o;
- and [```rag_metrics_with_ablation.ipynb```](https://anonymous.4open.science/r/from-MTEB-to-MTOB/rag_metrics_with_ablation.ipynb): calculating the metrics for ablation.
