# An Investigation of the Role of Virtual Reality and Its Impact on Student Engagement and Learning Retention

## Overview

This project explores the impact of **Virtual Reality (VR)** on **student engagement** and **learning retention** using a dataset sourced from Kaggle. The notebook combines exploratory data analysis (EDA), statistical testing, and machine learning classification techniques to evaluate how VR affects educational outcomes.

The analysis aims to uncover which factors within VR-based learning environments most significantly influence engagement and retention — supporting the broader study on VR’s pedagogical effectiveness in South Africa.

---

## Dataset

**Source:** [Kaggle – Virtual Reality in Education Dataset](https://www.kaggle.com/datasets/duyqun/virtual-reality-in-education-dataset)

**Dataset Description:**
The dataset includes various variables related to VR learning activities such as:

* `Immersion_Level`
* `Interactivity_Score`
* `Cognitive_Engagement`
* `Learning_Retention`
* `Stress_Level`
* `Enjoyment_Score`
* `Usage_Frequency`
* Demographic and contextual variables (e.g., `Age`, `Gender`, `Field_of_Study`)

The dataset was cleaned and preprocessed for machine learning analysis to predict and evaluate relationships between **VR experience quality** and **student learning outcomes**.

---

## Libraries Used

The notebook utilizes several Python libraries for data handling, visualization, and machine learning:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from mlxtend.plotting import plot_decision_regions
import statsmodels.api as sm
```

---

## Project Steps

### 1. **Data Import and Cleaning**

* The dataset is loaded and checked for missing or inconsistent values.
* Encoding is applied to categorical variables.
* Features are standardized using `StandardScaler`.

### 2. **Exploratory Data Analysis (EDA)**

* Distribution plots, correlation heatmaps, and summary statistics are used to understand data structure.
* Visualizations identify patterns between **immersion, engagement,** and **retention**.

### 3. **Model Building**

Several supervised learning models are implemented to predict engagement and retention outcomes:

* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **K-Nearest Neighbors (KNN)**

### 4. **Model Evaluation**

Each model’s performance is evaluated using:

* **Accuracy score**
* **Confusion matrix**
* **ROC curve & AUC**
* **Classification report**

### 5. **Findings and Insights**

* Higher levels of **interactivity** and **immersion** strongly correlate with improved engagement and retention.
* **Cognitive load** and **stress level** influence retention negatively if VR content is poorly designed.
* Logistic Regression and Random Forest models produced the most reliable results in predicting engagement outcomes.

---

## Key Research Questions

1. How does VR boost student engagement?
2. How does VR boost learning retention?
3. What factors (immersion, interactivity, stress) most significantly impact these outcomes?

---

## Results Summary

* **VR engagement** increases when activities are interactive, immersive, and meaningful.
* **Retention** improves when learning includes visual, kinaesthetic, and emotional elements.
* **Technical limitations** and **digital inequality** can reduce the benefits of VR in educational contexts.

---

## Theoretical Alignment

Findings align with **constructivist learning theory**, emphasizing that active, experiential learning enhances understanding and retention. The results also support literature by Lin et al. (2024), Suhag (2024), and AlAli & Wardat (2024) on VR’s potential to enhance cognitive engagement.

---

## Running the Notebook

1. Clone or download this repository.
2. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
3. Ensure all dependencies are installed:

   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn mlxtend statsmodels
   ```
4. Run all cells sequentially.


---

* Student Number : ST1029778
* Student Name: Khatija Essa
