
---

## 1. 📊 Data Preprocessing & Exploratory Data Analysis (EDA)

#### The goal of this phase was to:
- Understand feature distribution
- Detect skewness and outliers
- Identify multicollinearity
- Justify feature selection
- Prepare data for robust and interpretable modelling

### 📁 Data Overview
Each row in the dataset represent California census block group.
Features include:
- ```MedInc``` – Median income (in $10,000)
- ```HouseAge``` – Median house age (years)
- ```AveRooms``` – Average/Mean number of rooms per household
- ```AveBedrms``` – Average number of bedrooms per household
- ```Population``` – Population of the block group
- ```AveOccup``` – Average/Mean household occupancy
- ```Latitude``` – Geographic latitude
- ```Longitude``` – Geographic latitude

Target variable:

- ```MedHouseValue``` – Median house value (in $100,000)

---

### 📈 Feature Distribution Analysis (Histograms)
Histograms were generated to understand how each feature is distributed

#### Histograms
![Histograms](core/static/core/images/screenshots/histogram.png)

#### Why histograms were used?
Helped identify:
- Skewness (left or right)

- Presence of outliers

- Artificial caps or truncation

- Engineered or ratio-based features

#### Key Observation
Most features exhibit **right-skewed distributions**, particularly `MedInc`, `Population`, `AveRooms`, and `AveOccup`, indicating the presence of outliers and the need for feature scaling. `HouseAge` shows a clear upper cap around 52 years, reflecting a dataset constraint rather than a natural boundary. Several features are engineered ratios, which informed later preprocessing and feature selection decisions.


---

### 🔗 Correlation Analysis & Multicollinearity

A correlation heatmap was generated to identify linear dependencies between features.

#### Correlation Heatmap
![Correlation Heatmap](core/static/core/images/screenshots/histogram.png)

#### Why Correlation analysis?
Highly correlated features:
- Introduce redundancy

- Increase variance in linear models

- Reduce interpretability

- Add unnecessary complexity

Correlation analysis was performed only on input features, excluding the target variable.

#### Key Observations
- `AveRooms` and `AveBedrms` showed strong positive correlation, indicating redundant information.
- `AveBedrms` was removed during preprocessing, as `AveRooms` provides broader explanatory power.
- `Latitude` and `Longitude` exhibited strong negative correlation; however, both were retained because they encode distinct spatial dimensions essential for geographic modeling.


---

### 🌍 Geographic Visualization

A scatter plot of longitude versus latitude was generated to visualize spatial distribution.

### Scatter Plot
![Scatter plot](static/screenshots/demo2.png)

#### Insights:
- Clear diagonal structure reflecting California’s geography

- Dense clusters around major urban areas

- Confirms that spatial features are meaningful predictors

- This visualization further justified retaining both geographic coordinates despite their high correlation.


---


### ✅ Final Feature Selection

- ```MedInc```
- ```HouseAge```
- ```AveRooms```
- ```Population```
- ```AveOccup```
- ```Latitude```
- ```Longitude```

---

## 2. 📊 Model Training, Testing, and Evaluation.


---

---

