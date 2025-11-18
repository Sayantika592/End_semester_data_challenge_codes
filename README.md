Name: Sayantika Chakraborty    Roll No.: ME22B190

# AI Response Scoring Project

This project builds models to automatically predict the quality score (0–10) of AI-generated responses using text embeddings, feature engineering, and machine learning.

The evaluation metric used in this challenge is **RMSE (Root Mean Squared Error)**.  
RMSE measures how close the model’s predicted score is to the actual ground-truth score.  
A **lower RMSE value indicates better prediction accuracy**.

---

### Objective

Given:

- a user prompt  
- the AI response  
- the system prompt  
- the evaluation metric name  

The goal is to predict a numerical score that reflects how well the response satisfies the metric.

---

### Dataset Fields

| Field | Meaning |
|-------|---------|
| `user_prompt` | What the user asked |
| `response` | The model-generated reply |
| `system_prompt` | Instructions guiding the model |
| `metric_name` | The scoring rule being applied |
| `score` | Ground truth score (training only) |

The test set contains the same fields **except the score**, which must be predicted.

---

### Feature Engineering

To build strong predictors, multiple feature types were combined:

- **Sentence-transformer embeddings** of the full text  
- **Metric name embeddings**  
- **Pairwise similarity features** (cosine similarity, Euclidean distance, correlations, vector norms, angular distance)  
- **Meta-features**, such as:
  - length of prompt and response  
  - number of digits and question marks  
  - response-to-prompt length ratio  

These features help the model understand both **semantic meaning** and **text structure**.

---

### Models Used

Two different modeling approaches were tested and compared.

#### LightGBM Model

- Trained using stratified K-fold cross-validation  
- Used sample weighting to address label imbalance  
- Early stopping was applied  
- Best result among tested methods

**Validation RMSE:** **~3.625**

LightGBM performed well because tabular + embedding feature spaces often favor tree-based gradient boosting models.

---

#### Neural Network (PyTorch)

- Fully connected regression architecture  
- Batch normalization + ReLU activation  
- Dropout for regularization  
- Trained with MSE loss and evaluated using RMSE  

**Validation RMSE:** ~3.908

The neural network trained smoothly and generalized reasonably well, but did not surpass LightGBM for this dataset size and feature format.

---

### Model Performance Comparison

| Model | Validation RMSE | Notes |
|--------|---------------|-------|
| LightGBM | **≈ 3.625** | Best performing model |
| Neural Network | ≈ 3.908 | Stable, but underperformed |

---

### Kaggle Evaluation Metric

Kaggle evaluates models using **Root Mean Squared Error**:

\[
RMSE = \sqrt{\frac{1}{N} \sum (y_{true} - y_{pred})^2}
\]

RMSE penalizes large errors more strongly, so predictions must be close to the true values — not just directionally correct.

---

### Output Format

Submission file structure:

| ID | score |
|----|-------|
| integer index | predicted score between 0 and 10 |

Predictions were clipped to remain within the valid scoring range: **`[0, 10]`**.

---

### Possible Improvements

Future enhancements may include:

- Hyperparameter search (Optuna / Bayesian Optimization)  
- Model stacking or blending  
- Transformer fine-tuning instead of static embeddings  
- Contrastive or metric-aware representation learning  

---

### Summary

This project demonstrates how combining:

- semantic embeddings  
- handcrafted features  
- structured learning approaches  

can effectively predict scoring behavior for AI-generated text.

The LightGBM-based pipeline delivered the strongest results for this dataset and modeling setup.


