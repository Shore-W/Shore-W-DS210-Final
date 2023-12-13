"# Shore-W-DS210-Final" 

## DS 210 Final Project: K-means Clustering for Cholera Analysis
### Project Overview
### Title: Designing and Performing a K-means Clustering Algorithm to Analyze John Snow’s Cholera 1855 Report

*Author: Will Shore*

Date: 12/11/23

#### Introduction
For my DS 210 final project, I delved into the analysis of a dataset collected by John Snow in 1855 to track the spread of cholera deaths across London during a cholera epidemic. This historical dataset comprises 1,852 buildings in the Soho area, each associated with varying cholera death counts. The goal of my project is to employ a K-means clustering algorithm to predict deaths across clusters of houses.

#### Methods
*Processing*
I processed the data in Rust using a graph.rs module, converting the CSV file into a vector of records. The key features include ID, deaths (cholera-related and others), distances to significant points, and coordinates. Focusing on relevant features—distance to pestfield, sewer, and broad street pump—I scaled the data using min-max scaling for normalization.

*Clustering*
In the kmeans.rs module, I implemented the K-means clustering algorithm. After splitting the data into training and testing sets, the program identifies the best clustering iteration by calculating silhouette scores. The best iteration's characteristics, including cluster averages, are stored.

*Plotting*
Utilizing Rust plotters, I generated 2D and 3D graphs to visualize clustering results. The colors representing clusters are randomly generated, providing flexibility for any number of clusters. Circles represent nodes, and triangles indicate cluster centroids.

*Tests*
To validate functionality, three tests are incorporated in main.rs. The first checks the plotting function, the second evaluates the K-means algorithm on a dataset, and the third ensures proper dataset partitioning.

#### How to Run
Provide the program with the CSV file path (e.g., "john snow").
Specify the number of clusters and iterations for K-means.
Adapt plot.rs to match the project folder location.
Run the program, and PNG files will be generated with results. Save copies if needed.
