# Linear Regression Project

This project demonstrates the implementation of a simple Linear Regression model for predicting house prices based on several features. The dataset used in this project contains information about house prices and various attributes such as house age, distance to the nearest MRT station, number of convenience stores, latitude, and longitude.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Evaluation](#evaluation)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project locally, you'll need Python 3 and the following libraries installed:

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

Clone this repository to your local machine using:

```bash
git clone https://github.com/Marcocholla01/ML-Linear-Regression.git
```

## Usage
After installing the required dependencies and cloning the repository, navigate to the project directory and run the main script:

```bash
python main.py
```
This will train the Linear Regression model on the provided dataset and evaluate its performance.

## Data
The dataset used in this project (`real_estate.csv`) contains the following columns:

- `X1 transaction date`: Transaction date
- `X2 house age`: Age of the house
- `X3 distance to the nearest MRT station`: Distance to the nearest MRT station
- `X4 number of convenience stores`: Number of convenience stores in the area
- `X5 latitude`: Latitude of the location
- `X6 longitude`: Longitude of the location
- `Y house price of unit area`: House price per unit area (target variable)

## Models
The project uses a `simple Linear Regression model` to predict house prices based on the features provided in the dataset.

## Evaluation
The performance of the Linear Regression model is evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Output
These are some of the sample outputs recorded

### During Exploratory Data Analysis (EDA)
#### 1. Summary Statictics

![Summary Statictics](/output/summaryStatistics.png)

#### 2. Histograms of numerical features

![Histograms of numerical features](/output/numericalFeatures.png)

### During Visualizations

#### 1. Plotting predicted vs actual values
![Plotting predicted vs actual values](/output/LinearGraph.png)

#### 2. Plotting distribution of residuals
![Plotting distribution of residuals](/output/LinearBar.png)



## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt) file for details.