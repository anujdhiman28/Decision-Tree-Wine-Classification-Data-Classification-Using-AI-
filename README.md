#  Decision Tree Classification on Wine Dataset

A machine learning project developed using **Python** and **Scikit-learn** that classifies wine samples into different classes using the **Decision Tree Classifier**. The project demonstrates the complete machine learning workflow, from loading the dataset to evaluating model performance and visualizing accuracy across different tree depths.

---

##  Features

* Uses the built-in **Wine Dataset** from Scikit-learn
* Data exploration and preprocessing
* Train-Test Split (80:20)
* Decision Tree Classification
* Model evaluation using:

  * Accuracy Score
  * Precision
  * Recall
  * F1-Score
  * Classification Report
* Accuracy visualization for different `max_depth` values

---

##  Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib

---

##  Dataset Information

* **Dataset:** Wine Dataset
* **Samples:** 178
* **Features:** 13
* **Classes:** 3

---

##  Model Performance

* **Test Accuracy:** **94.44%**

The model achieved high classification accuracy on the test dataset while maintaining strong precision, recall, and F1-score across all three wine classes.

---

##  Accuracy Visualization

The project includes a graph showing how the model's accuracy changes with different values of `max_depth`.

**Observation:**

* Accuracy improves significantly as the tree depth increases.
* The model reaches approximately **94.44% accuracy** from **max_depth = 3** onwards.
* Increasing the depth beyond 3 does not significantly improve performance, indicating an optimal tree depth for this dataset.

---

##  Output

The program displays:

* Dataset Shape
* Feature Names
* Target Classes
* First Five Records
* Target Distribution
* Test Accuracy
* Classification Report
* Accuracy vs. Maximum Tree Depth Graph

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/DecisionTree-Wine-Classification.git
```

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib
```

Run the project:

```bash
python decision_tree.py
```

---

##  Learning Outcomes

* Decision Tree Classification
* Data Preprocessing
* Model Training & Testing
* Performance Evaluation
* Hyperparameter Analysis
* Data Visualization
* Supervised Machine Learning

---

##  Future Enhancements

* Decision Tree Visualization
* Hyperparameter Tuning using GridSearchCV
* Confusion Matrix
* Feature Importance Visualization
* Comparison with Random Forest, SVM, and KNN
* Streamlit Web Application

---

##  Sample Output

* Console output displaying dataset information, classification report, and accuracy.
* Line graph showing **Decision Tree Test Accuracy by Maximum Depth**.
