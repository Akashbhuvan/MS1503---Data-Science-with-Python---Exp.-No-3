# Experiment 3 : Retail Sales and Profit Analysis Using Pandas

## Aim
To create a sample retail dataset, identify and handle missing values (`NaN`/`None`) and duplicate records, and compute summary statistics using Pandas in Python.

## Algorithm

1. **Start** the program.
2. **Import Library**: Import the `pandas` library as `pd` for tabular data cleaning and analysis.
3. **Initialize Raw Retail Dataset**: Create a Python dictionary `data` containing retail records (`Order_ID`, `Product`, `Region`, `Sales`, `Discount`, `Profit`) with deliberate missing (`None`) entries and duplicate rows.
4. **Construct DataFrame & Inspect**:
   * Convert `data` into a Pandas DataFrame `df` using `pd.DataFrame(data)`.
   * Print the original dataset and its initial dimensions (`df.shape`).
5. **Detect Missing Values**:
   * Count missing entries per column using `df.isnull().sum()`.
6. **Impute Missing Values**:
   * Replace missing numerical values in `Sales`, `Discount`, and `Profit` with their respective column means using `.fillna(df[col].mean())`.
   * Replace missing categorical values in `Region` with the column mode (most frequent category) using `.fillna(df["Region"].mode()[0])`.
7. **Remove Duplicate Records**:
   * Drop exact duplicate rows from `df` using `df.drop_duplicates()`.
8. **Display Cleaned Dataset & Summary Statistics**:
   * Print the cleaned DataFrame, updated dimensions (`df.shape`), and descriptive statistical metrics (`df.describe()`).
9. **Stop** the program.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation & Inspection** | `pd.DataFrame()` | Converts a Python dictionary into a labeled 2D tabular DataFrame |
| | `df.shape` | Returns the `(rows, columns)` dimensions before and after cleaning |
| | `df.isnull().sum()` | Counts the number of missing (`NaN`/`None`) values in each column |
| **Missing Value Imputation** | `df[col].fillna()` | Replaces missing values with a specified replacement value |
| | `df[col].mean()` | Calculates the arithmetic mean for numerical column imputation |
| | `df[col].mode()[0]` | Retrieves the most frequent category for categorical column imputation |
| **Deduplication & Summary** | `df.drop_duplicates()` | Identifies and removes duplicate rows from the DataFrame |
| | `df.describe()` | Generates count, mean, standard deviation, min, quartiles, and max for numerical features |

## Program
# Experiment 3: Regression Techniques in Business Data(Simple Linear Regression)

##  Aim
To implement Simple Linear Regression in Python to study and predict the relationship between two business-related variables.

## Dataset
A sample business dataset representing the relationship between **Advertising Expenditure** (in Lakhs) and **Sales Revenue** (in Crores). The goal is to analyze how changes in advertising investment affect sales performance.

| Index | Advertising (Lakhs) | Sales (Crores) |
| :---: | :---: | :---: |
| 0 | 18 | 12 |
| 1 | 22 | 18 |
| 2 | 25 | 20 |
| 3 | 30 | 24 |
| 4 | 35 | 28 |
| 5 | 40 | 35 |
| 6 | 45 | 38 |
| 7 | 50 | 40 |
| 8 | 55 | 42 |
| 9 | 60 | 45 |

## Algorithm / Steps Followed
1. Import required Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Create and load the dataset into a Pandas DataFrame.
3. Display the dataset.
4. Select independent ($X$) and dependent ($Y$) variables and reshape $X$ into a 2D array.
5. Build a `LinearRegression` model using `sklearn.linear_model`.
6. Fit the model and calculate the intercept ($a$) and slope ($b$).
7. Predict $Y$ values for given $X$ values and evaluate performance using the $R^2$ score.
8. Plot the actual data points alongside the fitted regression line and predict sales for a new advertising budget.

##  Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation / Handling** | `np.array()` | Create numerical arrays for regression input data |
| | `pd.DataFrame()` | Create or manage tabular data for regression |
| **Model Building (Scikit-learn)** | `LinearRegression()` | Initialize the Simple Linear Regression model |
| | `model.fit()` | Train the linear regression model using $X$ and $Y$ values |
| | `model.predict()` | Predict $Y$ values for given $X$ inputs using the fitted model |
| | `model.score()` | Calculate model performance ($R^2$ score) |
| **Visualization (Matplotlib)** | `plt.scatter()` | Plot scatter points (actual data points) |
| | `plt.plot()` | Plot regression line (predicted relationship) |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.show()` | Configure and display the regression plot |

##  Program & Output
```# REGRESSION TECHNIQUES IN BUSINESS DATA
# (Simple Linear Regression)

# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
```
## Input 1
```
# Step 2: Create Dataset
data = {
    'Advertising': [18, 22, 25, 30, 35, 40, 45, 50, 55, 60],
    'Sales': [12, 18, 20, 24, 28, 35, 38, 40, 42, 45]
}

# Create DataFrame
df = pd.DataFrame(data)

print("----- Dataset -----")
print(df)
```
## Output 1
```
----- Dataset -----
   Advertising  Sales
0           18     12
1           22     18
2           25     20
3           30     24
4           35     28
5           40     35
6           45     38
7           50     40
8           55     42
9           60     45
```
## Input 2 
```
# Step 3: Prepare Input (X) and Output (Y) Variables
X = np.array(df['Advertising']).reshape(-1, 1)
Y = np.array(df['Sales'])

# Step 4: Build and Train the Linear Regression Model
model = LinearRegression()
model.fit(X, Y)

# Step 5: Display Model Coefficients
print("\nIntercept (a):", model.intercept_)
print("Slope (b):", model.coef_[0])
```
## Output 2
```
Intercept (a): 0.5811563169164877
Slope (b): 0.7794432548179872
```
## Input 3
```
# Regression Equation
print("\nRegression Equation: Y = {:.2f} + {:.2f}X".format(
    model.intercept_, model.coef_[0]
))
```
## Output 3
```
Regression Equation: Y = 0.58 + 0.78X
```
## Input 4
```
# Step 6: Predict Y for Given X Values
Y_pred = model.predict(X)

print("\nPredicted Sales Values:")
print(Y_pred)
```
## Output 4
```
Predicted Sales Values:
[14.6111349  17.72890792 20.06723769 23.96445396 27.86167024 31.75888651
 35.65610278 39.55331906 43.45053533 47.34775161]

```
## Input 5
```
# Step 7: Evaluate Model Performance
r2 = model.score(X, Y)

print("\nModel Performance:")
print("R2 Score = {:.4f}".format(r2))
```
## Output 5
```
Model Performance:
R2 Score = 0.9736
```
## Input 6
```
# Step 8: Visualize Regression Line
plt.figure(figsize=(8, 6))

plt.scatter(X, Y, color='blue', label='Actual Data Points')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')

plt.title("Simple Linear Regression: Advertising vs Sales")
plt.xlabel("Advertising Budget (in ₹ Lakhs)")
plt.ylabel("Sales (in ₹ Crores)")
plt.legend()
plt.show()
```
## Output 6
<img width="686" height="547" alt="image" src="https://github.com/user-attachments/assets/6cbd9dad-8d2e-45b5-8b96-df41b6d56b4e" />

## Input 7
```
# Step 9: Example Prediction
new_value = np.array([[45]])
predicted_sales = model.predict(new_value)

print("\nPredicted Sales for Advertising ₹45 Lakhs = {:.2f} Crores".format(
    predicted_sales[0]
))
```
## Output 7
```
Predicted Sales for Advertising ₹45 Lakhs = 35.66 Crores

```

##  Result

The Simple Linear Regression model was successfully implemented using Python to predict Sales based on Advertising Expenditure.
The regression equation obtained is:
                                      Y=0.58+0.78X
This model shows a strong linear relationship between advertising and sales, with an R
#Cleaning and Preparing Retail Data using Pandas

#Step 1: Create a sample retail dataset

#Step 2: Check for missing values

#Step 3: Fill missing numeric values with column mean

#Step 4: Fill missing Region with mode (most frequent value)

#Step 5: Remove duplicate records

#Step 6: Display summary statistics



## Output


## Result
The program successfully creates a retail dataset, identifies missing entries and duplicate rows, imputes missing values using mean and mode strategies, removes duplicates, and outputs a clean dataset ready for downstream analysis.

