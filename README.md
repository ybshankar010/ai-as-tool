# Agent-Based Ensemble Learning with LLMs

This project explores whether **Large Language Models (LLMs)** can serve as intelligent aggregators of weak classifiers, producing superior accuracy compared to traditional ensemble techniques (Random Forests, Logistic Regression, LightGBM, etc.).

We use [Qwen3](https://ollama.ai/library/qwen3), llama3.1, gpt-oss via [Ollama](https://ollama.ai) and [LangGraph](https://github.com/langchain-ai/langgraph) to implement a **Simple agent** that takes:

- A record (from the Adult Income dataset)
- Predictions from multiple base models
- Reasoning capabilities of the LLM

The LLM then decides the final class (`<=50K` or `>50K`) and explains its reasoning.

---

## 🔑 Features

- **Traditional ensembles**: Logistic Regression, Decision Tree, Random Forest, KNN, LightGBM, Dummy Classifier.
- **Agent-based ensemble**: An LLM (Qwen3:8b) that reads predictions + record and outputs the final decision.
- **Metrics**: Accuracy, Precision, Recall, F1, ROC AUC compared across models.
- **Explainability**: The agent provides reasoning for its final prediction.

---

## 📊 Results (Sample)

## Model Evaluation Results (Test Set: 2.5k records)

| Model                   | Accuracy  | Precision | Recall | F1        | ROC-AUC   |
| ----------------------- | --------- | --------- | ------ | --------- | --------- |
| **Agent**               | 0.743     | 0.485     | 0.800  | 0.604     | 0.897     |
| Logistic Regression     | 0.808     | 0.575     | 0.821  | 0.676     | 0.899     |
| Decision Tree (depth=5) | 0.847     | 0.776     | 0.524  | 0.626     | 0.876     |
| Random Forest (200)     | 0.847     | 0.737     | 0.584  | 0.652     | 0.900     |
| kNN (k=5)               | 0.825     | 0.670     | 0.560  | 0.610     | 0.847     |
| **LightGBM**            | **0.863** | 0.774     | 0.621  | **0.689** | **0.922** |

> Note: The agent is not optimized yet — future work is to improve prompting, batching, and weighting.

---

## ⚙️ Installation

### Prerequisites

- Ubuntu 22.04/24.04 (tested)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.ai) with GPU support recommended

### Setup

```bash
# Clone repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Pull Qwen model (quantized version recommended)
ollama pull qwen3:8b-instruct-q4_K_M
```

## 🚀 Running Experiments

### 1. Data preparation

```bash
Run Cells in data_preparation.ipynb
```

### 2. Generate baselines

```bash
Run Cells in Models.ipynb
```

### 3. Run agent-based ensemble

```bash
Run Cells in ensemble_weak_classifier.ipynb
```

## 🔮 Future Work

- Experiment with **weighted ensemble** (agent uses per-model validation accuracy as prior).
- Use **smaller Qwen3 models** (3B, 1.8B) for faster aggregation.
- Explore **Graph-of-Thoughts** or **Toolformer-style** aggregation prompts.
- Extend to other datasets (CIFAR-10, IMDB reviews).

## 🙌 Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Ollama](https://ollama.ai)
- [Qwen Models](https://huggingface.co/Qwen)
- [UCI Adult Income dataset](https://www.kaggle.com/datasets/uciml/adult-census-income)
