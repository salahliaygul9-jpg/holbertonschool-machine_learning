# Error Analysis

## Learning Objectives

- Understand key concepts in evaluating model performance.
- Learn how to create and interpret a confusion matrix.
- Understand different types of errors, metrics, and model evaluation techniques.
- Learn about bias, variance, irreducible error, and Bayes error.

## Confusion Matrix
A **confusion matrix** is a table that describes the performance of a classification model.  
- **Rows** represent actual (true) labels.  
- **Columns** represent predicted labels.  
- Each cell `[i, j]` shows the number of instances where the true label is `i` and the predicted label is `j`.

## Type I and Type II Errors
- **Type I Error (False Positive):** Predict positive but actual is negative.  
- **Type II Error (False Negative):** Predict negative but actual is positive.

## Sensitivity, Specificity, Precision, Recall
- **Sensitivity / Recall (True Positive Rate):** Measures how well positives are identified.  
  $$ \text{Sensitivity} = \frac{TP}{TP + FN} $$
- **Specificity (True Negative Rate):** Measures how well negatives are identified.  
  $$ \text{Specificity} = \frac{TN}{TN + FP} $$
- **Precision:** Measures how many predicted positives are actually positive.  
  $$ \text{Precision} = \frac{TP}{TP + FP} $$

## F1 Score
The **F1 score** is the harmonic mean of precision and recall:  
$$ F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

## Bias and Variance
- **Bias:** Error due to overly simplistic assumptions in the model.  
- **Variance:** Error due to model sensitivity to small fluctuations in the training set.  

### Irreducible Error
Error inherent in the data that cannot be reduced by any model.

### Bayes Error
The lowest possible error achievable by any classifier on a given problem. Approximated by evaluating an ideal classifier or using flexible models on large datasets.

## How to Create a Confusion Matrix
1. Collect true labels and predicted labels.  
2. Count how many times each true class was predicted as each predicted class.  
3. Fill these counts into a **matrix** of shape `(num_classes, num_classes)`.
