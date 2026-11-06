# Student Performance Analysis and Prediction System
## Project Presentation Slide Deck & Seminar Script

Use this outline to create your presentation slides (PowerPoint / Google Slides / Canva). Each slide includes the exact title, bullet points, recommended visual artifact from `outputs/graphs/`, and an **Evaluator Script** (what to say during the presentation).

---

### Slide 1: Title Slide
- **Title:** Student Performance Analysis and Prediction System Using Data Mining
- **Subtitle:** An Academic Mini-Project Demonstrating Preprocessing, K-Means Clustering, and Decision Tree Classification
- **Presenter:** [Your Name / Roll Number / Branch]
- **Subject:** Data Mining / Machine Learning Lab
- **Institution:** [Your College / Department Name]

> **Speaking Script:**  
> *"Good morning respected evaluators. Today I present our Data Mining mini-project titled 'Student Performance Analysis and Prediction System'. In this project, we implement both unsupervised and supervised data mining algorithms to analyze student academic behavioral records, discover natural cohorts, and predict whether a student is likely to pass or fail their final examination."*

---

### Slide 2: Problem Statement & Motivation
- **The Core Issue:** Final examination grades ($G3$) are known only at the end of the semester, leaving no room for corrective academic intervention.
- **The Objective:** Can early diagnostic assessments ($G1$, $G2$) and non-academic behavioral habits (study hours, past failures, absenteeism) predict final outcomes early?
- **Data Mining Target:** Formulate a binary classification target:
  - **Pass:** $G3 \ge 10$ (Out of 20)
  - **Fail:** $G3 < 10$
- **Target Leakage Safeguard:** Strictly isolate final grade $G3$ from predictive input features.

> **Speaking Script:**  
> *"Traditionally, educators identify failing students only after final marks are published. Our goal is to leverage data mining techniques to identify students at risk of failing early in the semester. To ensure mathematical validity and real-world applicability, we strictly prevented target leakage by completely removing the final grade G3 from the training feature matrix."*

---

### Slide 3: Dataset Overview (UCI Machine Learning Repository)
- **Dataset Source:** UCI Machine Learning Repository - Mathematics Student Dataset (`student-mat.csv`).
- **Instances:** 395 student records.
- **Original Attributes:** 33 features covering demographics, study time, attendance, family support, and grades.
- **Academic Grading Scale:** 0 to 20 points (standard Portuguese secondary grading scale).
- **Target Proportions:**
  - **Passing Students:** 265 (67.1%)
  - **Failing Students:** 130 (32.9%)
- **Data Cleaning:** Verified 0 missing values; Scikit-learn `SimpleImputer` integrated as a safeguard.

> **Speaking Script:**  
> *"We utilized the real-world benchmark Student Performance Dataset from the UCI Machine Learning Repository. It contains 395 instances with 33 attributes. The class distribution shows approximately 67% passing students and 33% failing students. We cleaned and prepared the data, ensuring no missing attributes compromise the pipeline."*

---

### Slide 4: Data Mining Pipeline & CRISP-DM Architecture
- **CRISP-DM Stages Followed:**
  1. **Data Ingestion:** Load semicolon-delimited CSV using Pandas.
  2. **Data Cleaning:** Impute missing attributes with `SimpleImputer` (median for numerical, mode for categorical).
  3. **Categorical Encoding:** One-Hot Encoding via `ColumnTransformer`.
  4. **Unsupervised Modeling:** Feature scaling with `StandardScaler` followed by K-Means clustering.
  5. **Supervised Modeling:** Stratified 80:20 split + Decision Tree Classifier using Information Gain (Entropy).
  6. **Deployment:** Model persistence using `Joblib` and deployment through an interactive Streamlit dashboard.

> **Speaking Script:**  
> *"Our project architecture strictly adheres to the industry-standard CRISP-DM methodology. We built a robust Scikit-Learn ColumnTransformer pipeline that encapsulates both imputation and one-hot encoding, fitted exclusively on training folds to prevent data snooping."*

---

### Slide 5: Exploratory Data Analysis (Key Findings)
- **Visuals to Insert:** `outputs/graphs/g3_distribution.png` and `outputs/graphs/pass_fail_count.png`
- **Key Observation 1:** Grade distribution exhibits a pseudo-normal curve centered around a mean of 10.42/20, with a notable spike at grade 0 (students who dropped out or were absent).
- **Key Observation 2:** 67.1% of students pass, demonstrating a natural 2:1 class balance.

> **Speaking Script:**  
> *"During our exploratory analysis, we analyzed the final grade distribution. The mean grade across the cohort is 10.42 points out of 20. The histogram also reveals a small cluster of zero scores corresponding to students who withdrew or missed the examination."*

---

### Slide 6: Factor Analysis: Study Habits & Historical Examination Marks
- **Visuals to Insert:** `outputs/graphs/studytime_vs_g3.png` and `outputs/graphs/g1_g2_vs_g3.png`
- **Study Time Impact:** Students studying $>5$ hours weekly (tiers 3 & 4) achieve significantly higher median grades with lower failure variance.
- **Predictive Power of Earlier Exams:**
  - $G1$ vs. $G3$: Pearson correlation $r = 0.80$
  - $G2$ vs. $G3$: Pearson correlation $r = 0.90$
- **Conclusion:** Period 2 exam ($G2$) is the strongest leading predictor of final academic success.

> **Speaking Script:**  
> *"As seen in these plots, weekly study time shows a direct positive shift in median academic achievement. Furthermore, previous examinations G1 and G2 display an extremely strong linear correlation of r = 0.80 and r = 0.90 with final grades, proving their validity as diagnostic predictors."*

---

### Slide 7: Unsupervised Learning: K-Means Student Clustering
- **Visuals to Insert:** `outputs/graphs/elbow_curve.png` and `outputs/graphs/cluster_distribution.png`
- **Input Features:** `studytime`, `failures`, `absences`, `G1`, `G2` (Standardized via `StandardScaler`).
- **Determining $k$:** The Elbow Method demonstrates a clear inflection point at $k=3$ (WCSS drop: $1975 \rightarrow 1430 \rightarrow 1157$).
- **Objective Cluster Cohort Profiling:**
  - **Cluster 1 (154 students - 39.0%):** *High Academic Achievers* (Mean $G3 = 14.40$, Mean Failures $= 0.08$)
  - **Cluster 0 (199 students - 50.4%):** *Moderate / Average Performers* (Mean $G3 = 8.43$, Mean Failures $= 0.15$)
  - **Cluster 2 (42 students - 10.6%):** *At-Risk / Needs Support* (Mean $G3 = 5.21$, Mean Failures $= 2.12$)

> **Speaking Script:**  
> *"Next, we applied unsupervised K-Means clustering. Using the Elbow Method, we determined k=3 as the optimal number of clusters. Rather than arbitrarily assigning labels, our code dynamically profiles the clusters based on their actual average grades. Cluster 1 represents High Achievers with an average of 14.4, while Cluster 2 highlights At-Risk students averaging 2.1 past failures and a failing mean mark of 5.2."*

---

### Slide 8: Supervised Learning: Decision Tree Classifier
- **Algorithm:** Decision Tree Classifier (`DecisionTreeClassifier`).
- **Splitting Criterion:** **Information Gain (Entropy)**:
  $$\text{Entropy}(S) = -\sum p_i \log_2(p_i)$$
- **Train/Test Partition:** 80% Training (316 students), 20% Testing (79 students) with Stratified Sampling.
- **Hyperparameter Constraints:** `max_depth=4` and `min_samples_leaf=5` to prevent overfitting and guarantee explainable tree rules.
- **Visual to Insert:** `outputs/graphs/tree_visualization.png`

> **Speaking Script:**  
> *"For supervised classification, we trained a Decision Tree using Information Gain / Entropy as taught in our Data Mining syllabus. We restricted tree depth to 4 levels. This prevents the tree from overfitting on training noise and yields clear, human-understandable decision rules."*

---

### Slide 9: Model Evaluation & Performance Metrics
- **Visuals to Insert:** `outputs/graphs/confusion_matrix.png` and `outputs/graphs/metrics_comparison.png`
- **Evaluation on Unseen Test Set ($N=79$):**
  - **Accuracy:** **87.34%** (69 of 79 correct)
  - **Precision (Pass):** **100.00%** (0 False Positives / Zero false alarms)
  - **Recall (Pass):** **81.13%** (43 of 53 passing students identified)
  - **F1-Score:** **89.58%** (High harmonic balance)
- **Confusion Matrix:** $TN = 26, FP = 0, FN = 10, TP = 43$.

> **Speaking Script:**  
> *"Our model achieved an overall accuracy of 87.34% on unseen test students. Notice that Precision is 100%—meaning whenever the model forecasted that a student would pass, they indeed passed, preventing dangerous false optimism. The F1-score stands at a balanced 89.58%."*

---

### Slide 10: Interactive Web Dashboard (Streamlit Demo)
- **Framework:** Streamlit (Python).
- **Core Modules:**
  1. **Home Dashboard:** High-level metrics, KPI summary cards, cohort distributions.
  2. **Data Exploration:** Dynamic interactive filtering, data audits, EDA charts.
  3. **Student Clustering:** Interactive Elbow slider and scatter plot inspection.
  4. **Performance Prediction:** Live inference form with probability breakdown and decision explanations.
  5. **Model Evaluation:** Real-time Confusion Matrix and Decision Tree visualizer.
  6. **About Project:** Syllabus mapping and technical references.

> **Speaking Script:**  
> *"To make this system practically accessible to educational administrators, we built an interactive web dashboard in Streamlit. It allows educators to explore student data, view clustering profiles, and enter an individual student's details to receive an instant Pass/Fail prediction with confidence probability."*

---

### Slide 11: Academic Conclusions & Data Mining Insights
1. **Predictive Confirmation:** Early exam marks ($G1$, $G2$) are the most decisive factors in final academic achievement.
2. **Behavioral Influence:** Weekly study time $>5$ hours significantly mitigates failure probability, while past course failures serve as primary risk indicators.
3. **Synergy of Unsupervised + Supervised:** K-Means clustering segments student populations without label bias, while Decision Trees provide explicit, interpretable rules.
4. **Target Leakage:** Completely avoided by removing $G3$, validating the model for real-world deployment.

> **Speaking Script:**  
> *"In conclusion, this project demonstrates the complete data mining lifecycle. We confirmed that diagnostic exam marks combined with behavioral habits provide strong predictive signals, allowing educators to intervene well before final examinations."*

---

### Slide 12: Future Enhancements & Thank You
- **Multi-Class Classification:** Predict letter grades ($A, B, C, D, F$).
- **Continuous Regression:** Predict exact final marks ($0 - 20$) using Random Forest or Gradient Boosting Regressors.
- **Ensemble Benchmarking:** Compare Decision Trees with Random Forests and XGBoost.
- **Institutional Deployment:** Integration with University Learning Management Systems (LMS).
- **Questions & Discussion:** Thank you to the evaluators.

> **Speaking Script:**  
> *"In future iterations, we plan to extend this binary classifier into a continuous grade regressor and benchmark against ensemble techniques. Thank you respected evaluators. I am now open to your questions."*

