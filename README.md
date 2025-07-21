# Cyberattack Detection with Random Forest, Apache Spark, and MongoDB

This project implements a cyberattack detection system using a Random Forest classifier trained with 100 decision trees. The system integrates Apache Spark for large-scale data processing, MongoDB for structured data storage, and Python for automation and performance evaluation. The solution was built using a labeled dataset from Kaggle.

## Project Overview

- **Goal**: Detect cyberattacks in network traffic using machine learning.
- **Model**: Random Forest with 100 decision trees.
- **Tools**: Apache Spark (Java), MongoDB, Python, Jupyter.
- **Dataset**: UNSW-NB15 (Kaggle)

## Technologies Used

- **Apache Spark (Java)**: Distributed data processing and model training.
- **MongoDB + Compass**: NoSQL database for storing preprocessed and prediction data.
- **Python**: Automation scripts for MongoDB setup and data ingestion.
- **Jupyter Notebook**: Visualization and evaluation of model accuracy, precision, recall, and F1-score.

## Dataset

This project uses the **UNSW-NB15 dataset**, available on Kaggle:  
[https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)

Please download and preprocess the dataset manually before running the pipeline.

## Repository Structure
```
cyberattack-detection-spark-rf/
│
├── spark-java/
│ ├── src/ # Spark Java source code
│ └── pom.xml # Maven dependencies
│
├── mongodb-automation/
│ └── mongodb_loader.py # Script for MongoDB setup and data upload
│
├── analysis/
│ └── metrics_visualization.ipynb # Accuracy and performance visualization
│
├── presentation/
│ └── cyberattack_detection.pptx # Project presentation slides
│
└── README.md ```


## Key Features

- **Random Forest Classifier**: Trained on labeled Kaggle data with 100 decision trees.
- **Spark Pipeline**: Parallel data transformation and classification.
- **MongoDB Integration**: Final predictions stored in a MongoDB collection.
- **Automation**: Python script to create the database and upload classification results.
- **Evaluation**: Accuracy and confusion matrix plotted in Jupyter.

## How to Run

1. **Java & Spark Setup**  
   - Build using Maven (`pom.xml`)
   - Run the Java code to train and evaluate the model using Spark

2. **MongoDB Setup**  
   - Ensure MongoDB is running (local or cloud)
   - Run `mongodb_loader.py` to create the database and upload processed data

3. **Performance Analysis**  
   - Open `metrics_visualization.ipynb` in Jupyter
   - Run all cells to view accuracy, precision, recall, F1-score, and plots

## Results

The model achieved high classification performance on a heavily imbalanced dataset, with all key metrics plotted in the Jupyter notebook.

## Report & Presentation

A full project presentation is included under `presentation/`. It covers:
- Problem statement and objectives
- Architecture and tools
- Design choices
- Results and visualizations

## License

This project is provided for academic and portfolio purposes.
