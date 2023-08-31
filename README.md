# Time-Series-Clustering-and-Classification
This work is based on my bachelorthesis in Hochschule Zittau/Görlitz. It aims to extract the time series that represent an ON/OFF event in the raw data, in which the change of active power will be recorded and used as the feature of clustering. Essentially this idea is on the basis of event-based NILM methods. 
## Time Series Clustering
The extracted time series will firstly be clustered according to the distance between each other. Given that two similar sequences (time series) may vary in speed, the traditional euclidean's distance is not suitable to mesaure the similarity between them. Taking shift in time or phase into account the DTW-Distance and its variant Soft-DTW will be used as   The normally used Clustering Methods are for the data points consisting of several unrelated features. 
