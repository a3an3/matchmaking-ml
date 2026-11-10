# Matchmaking – OkCupid Data Analysis

Machine learning project on the public
[OkCupid profiles dataset](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles)
(~60,000 anonymized dating profiles), based on the Codecademy capstone
*"Date-A-Scientist"*.

**Question:** Can a user's sex be predicted from their lifestyle answers and
the way they write their profile?

## Approach

- **Feature engineering** from raw profile answers: body type, drinking, smoking, drugs
  and education as ordinal codes; zodiac sign, religion and diet as categories
- **Text features** from the ten profile essays: total essay length and average word length
- **Preprocessing** in a scikit-learn Pipeline (MinMax scaling, one-hot encoding),
  fitted on the training set only
- **Five classifiers** compared on a stratified 80/20 train/test split:
  Decision Tree, Random Forest, k-Nearest Neighbors, Logistic Regression and SVM
- **Hyperparameter tuning**: depth sweep for the decision tree, k sweep for KNN (best k = 20),
  grid search for the SVM (best: RBF kernel, C = 100, γ = 0.01)

## Results

15,496 complete profiles (59 % male, 41 % female), stratified 80/20 train/test split.

| Model | Train accuracy | Test accuracy | Balanced accuracy |
|---|---|---|---|
| Baseline (always predict "male") | 58.6 % | 58.6 % | 50.0 % |
| Random Forest | 77.3 % | **67.2 %** | **63.9 %** |
| Decision Tree (depth 6) | 68.4 % | 66.5 % | 62.9 % |
| SVM (RBF) | 67.5 % | 66.1 % | 62.6 % |
| Logistic Regression | 66.2 % | 65.4 % | 61.9 % |
| KNN (k = 20) | 67.7 % | 62.1 % | 60.0 % |

**Takeaway:** Lifestyle answers and essay statistics carry a real but modest signal:
the best model beats the majority-class baseline by about 9 percentage points.
Tree-based models work best, and train and test accuracy stay close, so the models generalize.

## Tech stack

Python · pandas · NumPy · scikit-learn

## Run

```bash
pip install pandas numpy scikit-learn
python main.py
```

Place `profiles.csv` from the dataset next to `main.py` before running.
