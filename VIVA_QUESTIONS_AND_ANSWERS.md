# Student Performance Analysis and Prediction System
## Master Viva Voce & Academic Defense Q&A Guide

This guide compiles the most frequently asked viva and oral evaluation questions asked by external examiners and professors for this specific Data Mining project.

---

### Category 1: Problem Definition & Data Preprocessing

#### Q1: What is the primary problem your project addresses?
**Answer:**
"Educational institutions typically observe final grades ($G3$) at the end of the academic year, when it is too late to intervene. Our system leverages earlier diagnostic examinations ($G1$, $G2$) along with demographic, lifestyle, and behavioral factors to discover student cohorts using unsupervised clustering and predict whether a student is at risk of failing ($G3 < 10$) using supervised decision trees early enough to take corrective actions."

#### Q2: What is Target Leakage, and how did you prevent it?
**Answer:**
"Target Leakage occurs when an input feature contains information about the target variable that would not naturally be available at the time of prediction.  
In our dataset, the target is binary `passed` derived from final grade $G3$ ($G3 \ge 10$). If we included $G3$ in the feature set $X$, the decision tree would split on $G3 \ge 9.5$, achieving artificial 100% accuracy while being completely useless in practice.  
**Prevention:** We strictly excluded $G3$ from the feature matrix $X$ and used only prior diagnostic exam marks ($G1$, $G2$) and non-academic variables."

#### Q3: Why did you use Scikit-Learn's `SimpleImputer` and `ColumnTransformer` inside a `Pipeline`?
**Answer:**
"Using a Scikit-Learn `Pipeline` ensures that preprocessing operations (such as calculating median imputation values or learning one-hot encoding categories) are learned exclusively from the training fold (`X_train`) and then applied to the test fold (`X_test`). This strictly prevents **Data Snooping / Information Leakage** across the train-test split boundary."

#### Q4: Why did you use Stratified Splitting for the train-test split?
**Answer:**
"In our dataset, passing students comprise ~67.1% (265 students) and failing students comprise ~32.9% (130 students). A standard random split might accidentally assign a disproportionate number of failing students to either train or test sets. `StratifiedShuffleSplit` ensures that both training (80%) and testing (20%) sets preserve the exact same class proportions."

---

### Category 2: Unsupervised Learning (K-Means Clustering)

#### Q5: What is the fundamental difference between Clustering and Classification?
**Answer:**
- **Clustering (Unsupervised):** The algorithm groups data points based solely on feature similarity (Euclidean distance) without any teacher signal or target labels. It finds natural patterns in data.
- **Classification (Supervised):** The algorithm is trained with both feature attributes ($X$) and ground-truth target labels ($y$) to learn a decision boundary mapping inputs to outputs.

#### Q6: Why did you standardize features using `StandardScaler` before running K-Means?
**Answer:**
"K-Means computes Euclidean distance:
$$d(p, q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$
If one attribute has a large range (e.g., `absences` ranging from 0 to 93) while another has a small range (e.g., `studytime` ranging from 1 to 4), the attribute with the larger scale dominates the distance calculation. `StandardScaler` standardizes each feature to have a mean of 0 and a variance of 1 ($\mu=0, \sigma=1$), giving every academic feature equal mathematical weighting."

#### Q7: How does the Elbow Method work mathematically?
**Answer:**
"The Elbow Method evaluates Within-Cluster Sum of Squares (WCSS or Inertia):
$$\text{WCSS} = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$$
As $k$ increases, WCSS naturally decreases because clusters become smaller. The 'elbow point' is the point of diminishing returns where increasing $k$ further yields only marginal decreases in inertia. In our experiments, $k=3$ produced the clear inflection point."

#### Q8: Did your code assume Cluster 0 was 'High Performing' and Cluster 1 was 'Low Performing'?
**Answer:**
"No. K-Means assigns cluster integer IDs arbitrarily depending on initial centroid placement. Our implementation programmatically calculates the actual average final marks ($G3$ and $G2$) for each cluster and ranks them dynamically. Cluster 1 was identified as High Achievers (mean $G3 = 14.40$), Cluster 0 as Moderate Achievers (mean $G3 = 8.43$), and Cluster 2 as At-Risk Students (mean $G3 = 5.21$, mean past failures $= 2.12$)."

---

### Category 3: Supervised Learning (Decision Tree)

#### Q9: What splitting criterion did you choose for the Decision Tree, and why?
**Answer:**
"We used **Information Gain (Entropy)** (`criterion='entropy'`).  
Entropy measures the impurity or disorder in a set of samples:
$$\text{Entropy}(S) = - \sum_{i=1}^c p_i \log_2(p_i)$$
At each internal node, the decision tree tests every possible attribute threshold and chooses the split that maximizes Information Gain:
$$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
This aligns directly with the classic ID3 and C4.5 algorithms taught in the academic Data Mining syllabus."

#### Q10: Why did you limit the tree depth (`max_depth=4`)?
**Answer:**
"An unconstrained decision tree will continue splitting until every leaf is pure ($\text{Entropy}=0$). On a training set of 316 records, this leads to **Overfitting**—the tree memorizes noise and idiosyncratic student records rather than generalizable rules. Setting `max_depth=4` and `min_samples_leaf=5` acts as pre-pruning, ensuring a robust model that generalizes well on unseen test data while remaining visually interpretable."

---

### Category 4: Model Evaluation & Metrics

#### Q11: Explain your Confusion Matrix results.
**Answer:**
"Tested on $79$ unseen students (20% holdout):
- **True Negatives ($TN = 26$):** Students who actually failed and were correctly predicted as Fail.
- **False Positives ($FP = 0$):** Students who failed but were incorrectly predicted as Pass (Zero false alarms).
- **False Negatives ($FN = 10$):** Students who passed but were predicted as Fail.
- **True Positives ($TP = 43$):** Students who passed and were correctly predicted as Pass.  
Overall Accuracy: $\frac{43 + 26}{79} = 87.34\%$."

#### Q12: Why is Precision 100% in your evaluation, and what does it mean practically?
**Answer:**
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{43}{43 + 0} = 1.00 \ (100\%)$$
"In practical educational terms, 100% precision means that whenever the model predicted that a student would **Pass**, that student indeed passed. The model never issued an unjustified pass prediction to an at-risk student."

#### Q13: What is the F1-Score, and why is it preferred over Accuracy when evaluating classifiers?
**Answer:**
"Accuracy can be misleading when classes are imbalanced (e.g., if 90% of students passed, a naive model predicting 'Pass' for everyone has 90% accuracy but zero diagnostic value).  
The $F_1$-score is the harmonic mean of Precision and Recall:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
Because the harmonic mean penalizes extreme imbalances between Precision and Recall, an $F_1$-score of **89.58%** confirms our classifier's balanced reliability across classes."

---

### Category 5: Architecture & Deployment

#### Q14: How does your Streamlit dashboard connect to the machine learning code?
**Answer:**
"The web application (`app.py`) loads the pre-trained Scikit-learn pipeline directly from `models/trained_model.pkl` using Joblib. When a user enters student data into the UI form, the application constructs a single-row Pandas DataFrame with the exact column names expected by the pipeline. The preprocessor handles imputation and one-hot encoding dynamically, and the classifier outputs the class prediction and probability."

#### Q15: What are the primary limitations of this project, and how can it be extended?
**Answer:**
- **Current Limitations:** The dataset is bounded to 395 records from two schools in Portugal, and the target is simplified to a binary outcome.
- **Future Work:** Extend to multi-class grade prediction ($A, B, C, D, F$), implement continuous regression to predict exact numeric grades ($G3 \in [0, 20]$), and evaluate ensemble models such as Random Forests and Gradient Boosting.

