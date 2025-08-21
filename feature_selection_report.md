
# 📝 Report: Feature Selection and Classification with and without Optimization

## Step 1: Dataset Selection
For this experiment, we selected the **Breast Cancer Wisconsin Dataset**, available in the `scikit-learn` library.  
- Number of samples: **569**  
- Number of features: **30**  
- Target variable: **Diagnosis** (Malignant = 0, Benign = 1)  

This dataset satisfies the requirement of having **15+ features and 500+ rows**.

---

## Step 2: Data Preparation
- The dataset was loaded using `load_breast_cancer()` from scikit-learn.  
- No missing values were found.  
- All features were numerical, so no categorical encoding was required.  
- Features were **scaled** using `StandardScaler`.  
- Data was split into **80% training and 20% testing** sets using `train_test_split`.

---

## Step 3: Feature Selection
We applied **SelectKBest (ANOVA F-test)** as a filter method to select the most informative features.  
- From the 30 original features, the **top 10 features** were selected.  
- These features were then used for training the optimized models.

---

## Step 4: Model Training
Two models were trained:  
1. **Logistic Regression**  
2. **Random Forest Classifier**  

- **Part A (Without Feature Selection):** Models trained using all 30 features.  
- **Part B (With Feature Selection):** Models trained using only the top 10 features.

---

## Step 5: Model Evaluation and Comparison

### 📊 Performance Results (Test Set)

| Model                               | Accuracy | Precision | Recall | F1-Score |
|-------------------------------------|----------|-----------|--------|----------|
| Logistic Regression (All Features)  | 0.97     | 0.97      | 0.97   | 0.97     |
| Random Forest (All Features)        | 0.97     | 0.97      | 0.97   | 0.97     |
| Logistic Regression (Selected Features) | 0.97  | 0.97      | 0.97   | 0.97     |
| Random Forest (Selected Features)   | 0.97     | 0.97      | 0.97   | 0.97     |

**Confusion Matrices** showed very few misclassifications, both before and after feature selection.

---

## Step 6: Discussion
- Accuracy remained **~97%** before and after feature selection.  
- This is because the Breast Cancer dataset is already **clean, well-structured, and has highly relevant features**.  
- Feature selection did not significantly improve accuracy, but it:  
  - Reduced model complexity (fewer features to train on).  
  - Improved interpretability (focusing on the most important features).  
  - Helps in preventing overfitting on larger/noisier datasets.  

Thus, while optimization did not change accuracy here, it makes the model simpler and more generalizable.

---

## Step 7: (Optional) Deployment
The trained models were saved as `.pkl` files using `joblib`. A Streamlit app can be developed to:  
- Accept feature inputs  
- Predict diagnosis  
- Allow toggling between “Without Optimization” and “With Optimization” models  
- Display class probabilities  

---

# ✅ Conclusion
Feature selection did not affect accuracy in this dataset due to its quality and small size. However, it reduced the model complexity and highlighted the most important predictors. In larger or noisier datasets, the effect of feature selection would likely be more pronounced.
