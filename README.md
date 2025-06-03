

# Email Spam Filtering using KNN and LSVC

This project implements two separate machine learning models — **K-Nearest Neighbors (KNN)** and **Linear Support Vector Classification (LSVC)** — to detect spam emails based on word frequency features.

## 📁 Dataset

* The dataset contains **5,172 emails**.
* Each email is represented by **3,002 columns**:

  * **Column 1:** `Email No.` (identifier)
  * **Columns 2–3001:** Frequency counts of the **3,000 most common words** in all emails (after preprocessing)
  * **Column 3002:** `Prediction` label — `1` for spam, `0` for not spam

## 📊 Exploratory Data Analysis (EDA)

* Distribution of spam vs. non-spam emails
* Check for missing values
* Basic statistical summary of the dataset

## 🧠 Models Used

### 1. K-Nearest Neighbors (KNN)

* Trained using default hyperparameters
* Evaluated on accuracy, confusion matrix, and classification report
* Model saved as: `spam-KNN.sav`

### 2. Linear Support Vector Classification (LSVC)

* Trained with `C=0.1`, `max_iter=1500`, `tol=0.001`, `random_state=42`
* Grid Search performed for hyperparameter tuning:

  * Parameters: `C`, `max_iter`, and `tol`
* Final model saved as: `spam-LSVC.sav`

## 📈 Evaluation Metrics

* **Accuracy Score**
* **Confusion Matrix** (Visualized using Seaborn heatmap)
* **Classification Report** (Precision, Recall, F1-Score)

## 🔧 Tools and Libraries

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn (KNN, LinearSVC, GridSearchCV, evaluation metrics)
* Pickle (for saving models)

## 💾 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/email-spam-filtering.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook or Python script containing the model code.

4. (Optional) Use the saved `.sav` models to make predictions on new email data.

## 📌 Notes

* This project treats KNN and LSVC as **independent models**, not a hybrid.
* Word count vectors serve as features — no raw text data is processed.

## 📁 Files

* `emails.csv` – Preprocessed dataset
* `spam-KNN.sav` – Saved KNN model
* `spam-LSVC.sav` – Saved LSVC model
* `notebook.ipynb` or `script.py` – Contains full code


