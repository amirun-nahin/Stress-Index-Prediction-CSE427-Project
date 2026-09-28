# Early Detection of Academic Burnout: Predicting Student Stress Indices with Machine Learning 🎓

**Course:** CSE427: Machine Learning  
**Institution:** BRAC University  
**Group:** 05

## 📝 Project Overview

The primary goal of this project is to build a machine learning system capable of predicting a student's **Stress Index**, a continuous score on a 10-point scale. University students juggle heavy class loads, financial worries, and personal challenges, and many hesitate to ask for help because of stigma. Even when they stay silent, their daily patterns (attendance, study hours, assignment delays, grades) can reveal signs of mounting stress.

This project uses machine learning to move from "self-reporting and guesswork" to **data-driven early detection**, so that institutions can offer counseling *before* academic burnout sets in. Unlike a typical classification task, this is a **regression problem**: we predict *how much* stress a student is under, not just whether they are stressed.

## 📂 Dataset Description

The dataset consists of **10,000 student records** with **18 columns** (1 identifier + 16 independent variables + 1 target variable).

* **Target Variable:** `Stress_Index` (Continuous, range 1.66 – 9.18, mean 5.51)
* **Identifier (dropped):** `Student_ID`
* **Numerical Features (9):** `Age`, `Family_Income`, `Study_Hours_per_Day`, `Attendance_Rate`, `Assignment_Delay_Days`, `Travel_Time_Minutes`, `GPA`, `Semester_GPA`, `CGPA`
* **Categorical Features (7):** `Gender`, `Internet_Access`, `Part_Time_Job`, `Scholarship`, `Semester`, `Department`, `Parental_Education`

## 🔍 Exploratory Data Analysis (EDA)

We performed statistical profiling and visualization to understand the data logic:

1.  **Missing Values:** Exactly **5%** of the data was missing in four columns: `Family_Income` (500), `Study_Hours_per_Day` (500), `Parental_Education` (511), and the target `Stress_Index` (500). No duplicate rows were found.
2.  **Skewness:** Most features were symmetric, and the target itself was almost perfectly balanced (skew ≈ -0.04). However, `Family_Income` was heavily **Right-Skewed** (skew ≈ 2.83), and `Assignment_Delay_Days` was mildly right-skewed (≈ 0.70).
3.  **Outlier Detection:** Using **Boxplots**, we found a massive outlier in `Family_Income`: the maximum value (316,601) was over 8× the population mean (≈ 38,377).
4.  **Correlation:** A heatmap analysis showed that academic performance (`GPA`, `Semester_GPA`, `CGPA`) has the strongest relationship with stress (**≈ -0.54**): better grades, lower stress. Demographic and lifestyle features such as `Age`, `Family_Income`, and `Travel_Time_Minutes` showed **near-zero linear correlation** with stress, suggesting that non-linear models would be needed. The three GPA columns are also very strongly correlated with each other (≈ 0.96 – 0.99).

## 🛠 Data Pre-processing

To make the data machine-readable, the following steps were taken:

* **Cleaning:** Dropped the 500 rows with a missing `Stress_Index` (we never train on fabricated targets) and removed the `Student_ID` column. This left **9,500 records**.
* **Feature Engineering:** Created a composite **`Academic_Risk`** score (0 to 3) that adds one point for each behavioral red flag: attendance below 75%, study hours below 2 per day, and assignment delays above 5 days.
* **Imputation:**
    * *Skewed Numerical (`Family_Income`):* Filled missing values with the **Median**, then applied a **log transformation** (`np.log1p`) to fix the skewed curve.
    * *Normal Numerical:* Filled missing values with the **Mean**.
    * *Categorical:* Filled missing values with the **Mode** (most frequent).
* **Encoding:**
    * *Ordinal* (`Semester`: Year 1 → Year 4, `Parental_Education`: High School → PhD): **OrdinalEncoder** with the order defined explicitly to preserve rank.
    * *Nominal* (`Gender`, `Internet_Access`, `Part_Time_Job`, `Scholarship`, `Department`): **One-Hot Encoding** (`drop_first=True`) to avoid the Dummy Variable Trap.
* **Scaling:** Applied **StandardScaler** to the numerical features so that large-valued columns do not overpower smaller ones.
* **Leakage Prevention:** Everything above is wrapped in a scikit-learn **`ColumnTransformer`** inside a **`Pipeline`** with each model, so the preprocessing is fitted only on training data.

## 🧠 Models Implemented

We trained six regression models on an 80/20 train-test split (`random_state=42`). Every model was first trained with base settings, then tuned using **`RandomizedSearchCV`** (5-fold cross-validation, 20 iterations, scored with negative Mean Squared Error).

1.  **Random Forest Regressor:** An ensemble of decision trees whose predictions are averaged. Tuned: number of trees, max depth, min samples per split/leaf.
2.  **AdaBoost Regressor:** Builds trees sequentially, with each new tree focusing on the samples the previous ones got wrong. Tuned: number of trees, learning rate, loss function.
3.  **XGBoost Regressor:** An optimized gradient boosting algorithm. Tuned: number of trees, learning rate, tree depth, subsample ratio, column sample ratio.
4.  **K-Nearest Neighbors (KNN) Regressor:** Predicts stress by averaging the scores of the K most similar students. Tuned: number of neighbors, weighting scheme, distance metric (Manhattan/Euclidean).
5.  **Multi-Layer Perceptron (MLP) Regressor:** A neural network with hidden layers, weights, biases, and activation functions. Tuned: hidden layer sizes, activation function, regularization (alpha), initial learning rate.
6.  **Bayesian Ridge Regressor:** A probabilistic linear model with built-in regularization. Tuned: the four regularization hyperparameters (`alpha_1`, `alpha_2`, `lambda_1`, `lambda_2`).

## 📊 Results

Since this is a regression task, models were evaluated on **RMSE**, **MAE**, and **R² Score** on the held-out test set (1,900 students).

| Model (Tuned) | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **1.0085** | **0.8052** | **0.3539** |
| XGBoost | 1.0130 | 0.8093 | 0.3481 |
| AdaBoost | 1.0323 | 0.8299 | 0.3230 |
| Bayesian Ridge | 1.0326 | 0.8288 | 0.3226 |
| MLP (Neural Network) | 1.0338 | 0.8278 | 0.3211 |
| KNN | 1.0698 | 0.8486 | 0.2730 |

**Effect of hyperparameter tuning (R²):**

| Model | Base | Tuned |
| :--- | :--- | :--- |
| Random Forest | 0.3506 | 0.3539 |
| XGBoost | 0.3385 | 0.3481 |
| AdaBoost | 0.3230 | 0.3230 |
| Bayesian Ridge | 0.3226 | 0.3226 |
| MLP | 0.2394 | 0.3211 |
| KNN | 0.1740 | 0.2730 |

### Conclusion & Analysis
The **Tuned Random Forest Regressor** performed the best, with an average error (MAE) of about **0.81 points** on the 10-point stress scale and an **R² of 0.354**. XGBoost followed very closely, and the two tree-based ensembles clearly outperformed the rest. **KNN** performed the worst, which suggests that distance-based methods struggle with this mix of behavioral and categorical features.

Tuning helped the most for the models that were weakest out of the box (**MLP** and **KNN**), while it made almost no difference for AdaBoost and Bayesian Ridge. Residual analysis of the best model showed errors scattered evenly around zero, with no obvious systematic over- or under-prediction.

An R² of about 0.35 means the recorded academic, financial, and demographic habits explain roughly a third of the variation in student stress. The remaining variation is most likely driven by personal factors that the university does not record, such as family, health, and social life. The model is therefore best used as an **early-warning screening tool** to flag students who may need support, not as a precise measurement of anyone's mental state.

## 👥 Contributors

* **Amirun Nahin** (ID: 23201416)
* **Sultan Mohammad Farid** (ID: 23201107)
* **Muhtasim Fuad** (ID: 23201082)

## 🎓 Instructor
Submitted to **Rafeed Rahman**, Senior Lecturer, BRAC University.
