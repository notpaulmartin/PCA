# PCA
Principal Component Analysis (PCA) is a key method in Data Science and Machine Learning to reduce the number of variables (dimensions) while retaining as much information (variance) as possible.

In PCA we aim to find a set of orthogonal vectors, each being a Principal Component (PC), that explain the most variance.

[Link to Notebook](https://deepnote.com/@notpaulmartin/PCApproximation-PIgieYKmR5aaz3DVjLKqdg)

## Approximating the Principal Components
In this notebook I approximate the principal components by assigning the axes X, Y, Z, W, ... to the principal components and rotate the data points so that their variance is maximised along the axes (PCs). To maximise variance along the first component (X-axis), the points are allowed to rotate in any direction. To then maximise variance along the subsequent components (e.g. the Y-axis), the poinst are only allowed to be rotated in a way that their position on the X-axis remains constant.

Example in 3D:
To align the points along the X-axis, they are allowed to rotate around the Y and Z-axes. However, to align them along the Y-axis, they are only allowed to rotate around the X-axis, as that does not change their position _on_ the X-axis.

## But why?
While this has a complexity of O(dimension^3 x n x 1/precision), which is an increase from [approximately O(dimension^3)](https://alekhyo.medium.com/computational-complexity-of-pca-4cb61143b7e5) for the standard procedure, the benefit of this is that the individual components can be computed independently of one another.
