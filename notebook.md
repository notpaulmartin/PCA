# Intuitive PCApproximation

This is an exploration into Principal Component Analysis (PCA) with the aim to explain to myself (and potentially others) in an intuitive way how PCA works.

I managed to create an algorithm that, while not as efficient as typical implementations of PCA (e.g. [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)), still runs reasonably quickly and, most importantly, is rather easy to understand – at least so I think.


**What is PCA?**  
PCA is a form of dimensionality reduction very commonly used in Data Science and Machine Learning. Most datasets contain a very large amount of variables, which makes it difficult to identify trends. PCA helps with this, as it reduces the number of variables while retaining as much variance (information) as possible. Visually, PCA works by plotting the points and then finding n orthogonal lines that maximise the variance, as seen in the image below. The spread of the points along these lines represent the n new variables.  

_Example:_  
To reduce a dataset from two variables to one, we first plot the points on a 2D plane and then find a single line that maximises their variance.  
![](https://dinhanhthi.com/img/post/ML/dim_redu/pca-1.jpg)  
_(Taken from https://dinhanhthi.com/principal-component-analysis/)_


**My approximation**  
Since the axes in euclidean geometry are already, by definition, orthogonal to one another, we can simply rotate them to maximise the variance along them. In 3D, we would first rotate the X-axis in any direction to maximise variance along it and then rotate the Y-axis while keeping the X-axis fixed and the Y-axis perpendicular to it. Once the variance has been maximised along both axes, the new variables are the X and Y-axes. This can be scaled up to any number of input and output variables.

Since rotating the axes is not really feasible, my algorithm rotates the points instead, but the idea remains the same.


```
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

import plotly.express as px

from sklearn import decomponsition
from sklearn import preprocessing

import statistics as stats

import mgen  # Generating rotation matrices
```


```
np.set_printoptions(suppress=True)  # Don't use scientific notation when printing numpy arrays
```

# Data creation


```
# Function to normalize data
normalize = lambda arr: preprocessing.normalize([arr])[0]
```


```
def gen3dData(m, c, size=20, noiseFactor=0.3):
    return genNdData(n=3, size=size, noiseFactor=noiseFactor)
    # y=mx+c
    xs = np.random.uniform(size=size)
    ys = [m * x**2 + c + np.random.normal() * noiseFactor for x in xs]
    zs = [m * x**2 + c + np.random.normal() * noiseFactor for x in xs]
    return xs, ys, zs

# Generates n-dimensional data (n input variables)
def genNdData(n=3, size=20, noiseFactor=0.3):
    variables = np.zeros((n, size))
    variables[0] = xs = np.random.uniform(size=size)

    for dim in range(1, n):
        variables[dim] = [x**(dim+1) + np.random.normal() * noiseFactor for x in xs]

    variables = [normalize(xs) for xs in variables]

    return variables
```


```
def unzip(xs):
    return list( zip(*xs) )

# Same as transposing a matrix
transpose = lambda xs: unzip(xs)
```


```
xs, ys = genNdData(n=2, size=20, noiseFactor=0.01)
vectors2d = transpose([xs, ys])

sns.scatterplot(x=xs, y=ys)

"Variance: ", stats.variance(xs)
```




    ('Variance: ', 0.008650515309871048)




    
![png](notebook_files/notebook_7_1.png)
    


# PCA (sklearn)


```
# PCA
pca = decomposition.PCA(n_components=1)
X_reduced = pca.fit_transform(vectors2d)
pca.explained_variance_ratio_
pca.components_
```




    array([[-0.57955567, -0.81493265]])




```
# Plot PCA
xs_reduced = np.transpose(X_reduced)[0]
sns.stripplot(x=xs_reduced, jitter=0.03)
```




    <AxesSubplot:>




    
![png](notebook_files/notebook_10_1.png)
    


# My approximation

**Rotations**  
In 3D we can generally say that we rotate around an axis or vector. This, however, is particular about the three dimensions that we live in and cannot be transferred to other dimensions: In 2D we can't rotate around an axis and neither can we in higher dimensions.

When we say that we rotate _around_ an axis, what we are really saying is that we are keeping the points constant (invariant) along the axis. The element (pivot) that we are keeping invariant depends on the dimension we are rotating in: In 2D we are only keeping the pivot **point** constant; In 3D it is a pivot **line**; In 4D it would be a pivot **rectangle**, etc.

| Dimensions ||||||| Invariant Pivot |
| ------------- | ------------------ |
| 2D         ||||||| 0D Point |
| 3D         ||||||| 1D Line |
| 4D         ||||||| 2D Rectangle |
| 5D         ||||||| 3D Cuboid |
| 6D         ||||||| 4D Tessaract |
| ...        ||||||| ... |
  
No matter how many dimensions we are operating in, we are always rotating on a 2D plane. Such a plane is typically described by two orthogonal vectors.

**Example:**  
In the following code block the vector \[1 0 0] is rotated 90° around the Z-axis, which corresponds to a rotation by 90° on the plane spanning the X and Y-axes (XY-plane).


```
vectorToRotate = [1, 0, 0]

rotation_degrees = 90
rotation_radians = np.radians(rotation_degrees)

plane_v1 = np.array([1, 0, 0])
plane_v2 = np.array([0, 1, 0])

# Create two rotation matrices to compare
M1 = mgen.rotation_around_axis([0, 0, 1], rotation_radians)
M2 = mgen.rotation_from_angle_and_plane(rotation_radians, plane_v1, plane_v2)

# e-16 is a floating point error and can be considered 0
print(M1.dot(vectorToRotate))
print(M2.dot(vectorToRotate))
```

    [0. 1. 0.]
    [0. 1. 0.]


## PCA approximation in 2D

To find the optimal rotation we rotate the points in small steps. On each step, we rotate by 1° clockwise and by 1° anti-clockwise and choose the direction that increases the variance along the X-axis. If the variance cannot be increased, we're done.


```
# Rotate a single vector in 2D
# (In 2D there is only one plane to rotate on)
def rotate(vec, angle_deg):
    angle_rad = np.radians(angle_deg)
    M = mgen.rotation_from_angle(angle_rad)
    return M.dot(vec)

# Rotate a list of vectors in 2D
def rotateAll(vectors, angle_deg):
    angle_rad = np.radians(angle_deg)
    M = mgen.rotation_from_angle(angle_rad)
    return [M.dot(v) for v in vectors]
```


```
def xsOf(vectors):
    return list( list(unzip(vectors))[0] )

def ysOf(vectors):
    return list( list(unzip(vectors))[1] )
```


```
def plot1d(xs):
    sns.stripplot(x=xs, jitter=0.03)

def plot2d(vectors):
    xs, ys = unzip(vectors)
    return sns.scatterplot(xs, ys)
```


```
angle = 0
angle_change = 1

var = -1
var_new = 0

vectors = np.copy(vectors2d)

while var < var_new:
    var = var_new

    vectors_left = rotateAll(vectors, -angle_change)
    vectors_right = rotateAll(vectors, angle_change)

    var_left = stats.variance(xsOf(vectors_left))
    var_right = stats.variance(xsOf(vectors_right))

    if var_left < var_right:
        var_new = var_right
        angle += angle_change
        vectors = vectors_right

    elif var_right < var_left:
        var_new = var_left
        angle -= angle_change
        vectors = vectors_left

    else:
        break

print(f'Optimal rotation: {angle}° clockwise')
print(f'Variance along X-axis (1st PC): {var_new}')
```

    Optimal rotation: -54° clockwise
    Variance along X-axis (1st PC): 0.02544352613903337


#### Results
And now let's look at how the approximation compares to standard PCA implementations:


```
# Variance explained
xs, ys = unzip(vectors2d)
var_explained = var_new/(stats.variance(xs) + stats.variance(ys))

print( "Variance explained by approximation:", var_explained)
print( "Variance explained by PCA:", pca.explained_variance_ratio_)

print("Error: ", (var_explained - pca.explained_variance_ratio_)/ pca.explained_variance_ratio_)
```

    Variance explained by approximation: 0.9938083377771011
    Variance explained by PCA: [0.99390981]
    Error:  [-0.00010209]


Comparing the variance explained by the 1st Principal Component found by standard PCA and by the approximation, the error is roughly 0.0001% depending on the data. Not bad!

To extract the 1st Principal Component we can simply take the x-values (2nd PC is y-values) of the resulting vectors:


```
plot1d(xsOf(vectors))
```


    
![png](notebook_files/notebook_23_0.png)
    


## PCA approximation in 3D and higher

### Generating and visualising some sample data


```
def plot3d(vectors):
    xs, ys, zs = unzip(vectors)
    fig = px.scatter_3d(x=xs, y=ys, z=zs)
    fig.show()
```


```
# Generate data
xs, ys, zs = genNdData(n=3, noiseFactor=0.02)
vectors3d = transpose([xs, ys, zs])
```


```
df = pd.DataFrame(vectors3d)
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x7fd046267710>




    
![png](notebook_files/notebook_28_1.png)
    



```
plot3d(vectors3d)
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>                <div id="3381c407-a8b3-4039-9052-ee4a416db210" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("3381c407-a8b3-4039-9052-ee4a416db210")) {                    Plotly.newPlot(                        "3381c407-a8b3-4039-9052-ee4a416db210",                        [{"hovertemplate": "x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "", "scene": "scene", "showlegend": false, "type": "scatter3d", "x": [0.3170538275577871, 0.03599742589533022, 0.18310228022607872, 0.20686266943650333, 0.10920391715559873, 0.14974620996805285, 0.07418190649014772, 0.27338553686259304, 0.2908777722020196, 0.28072814583424616, 0.27061478613369927, 0.2843347405421928, 0.21797279584976958, 0.1885206125507925, 0.18863967683016636, 0.059433730547625255, 0.3270986893818226, 0.23192036092151364, 0.26082342206532544, 0.19696374900382824], "y": [0.3796213339767134, -0.01311979424981107, 0.13072015719271501, 0.169502527707228, 0.0332014762274169, 0.07981850296344743, 0.019707770211096112, 0.2966042975103325, 0.3121339930711105, 0.2978870580911336, 0.27487784767644463, 0.2895816849720594, 0.18222274108223319, 0.125757289694055, 0.13010347178219134, 0.023950427862069516, 0.421711295577326, 0.19795693156571156, 0.2474699289368103, 0.15392143800906777], "z": [0.4327818751491886, -0.008909777374420096, 0.07179561976995547, 0.1419536972176876, 0.03197769921979442, 0.06577899768921336, 0.010883238870661791, 0.2765987063912328, 0.31558257001420387, 0.2852324757960779, 0.2711483276074484, 0.30886250573753216, 0.13687467822112812, 0.08672678555727412, 0.09892052702620631, 0.028236160659652086, 0.4674651092348054, 0.16988353961083844, 0.24615160707897085, 0.10324462247446593]}],                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "scene": {"domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "xaxis": {"title": {"text": "x"}}, "yaxis": {"title": {"text": "y"}}, "zaxis": {"title": {"text": "z"}}}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('3381c407-a8b3-4039-9052-ee4a416db210');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


### Algorithm in 3D and higher dimensions

The approach here is mostly the same as in 2D:
- We first rotate the points in small steps. On each step we rotate clockwise and anti-clockwise, choosing the direction that increases variance along the X-axis until we reach a maxima. We do this on every plane where the target axis is not invariant. For the X-axis in 3D this would be the XY and XZ planes.
- Once we reached maxima along the X-axis for rotations on all relevant planes, we focus on the Y-axis and maximise the variance along it. In order to not undo our work on the X-axis, we ignore all rotations that would affect it (all planes that include the X-axis). Hence in 3D we can only rotate on the YZ plane, but not the YX plane.
- This is progressively done for every axis until the number of axes we have maximised equals the number of PCs wanted, or until there is only one axis remaining, as there are no planes left that the points may be rotated on to maximise the last axis. (n-1 degrees of freedom)

Once complete, the X-axis is the 1st PC, the Y-axis the 2nd, etc.

#### Helper functions to rotate vectors


```
# Convert two axes into orthogonal vectors that describe their plane
def planeToVectors(dimension, axis1, axis2):
    v1 = np.zeros(dimension)
    v2 = np.zeros(dimension)
    v1[axis1] = 1
    v2[axis2] = 1
    return v1, v2

# e.g. In 3D between X and Z axes
#  planeToVectors(3, 0, 2)
#  -> [1 0 0], [0 0 1]
```


```
# Perform a clock-wise rotation by [rotation_deg]° on a list of vectors on the given plane
def rotateOnPlane(plane_v1, plane_v2, rotation_deg, vectors):
    # Create rotation matrix
    rotation_rad = np.radians(rotation_deg)
    M = mgen.rotation_from_angle_and_plane(rotation_rad, plane_v1, plane_v2)

    # Rotate all vectors
    return [M.dot(v) for v in vectors]
```


```
# Tries rotating vectors in both directions on the plane described by the two axes.
# Chooses the direction that maximises the variance along the optimisation_axis.
# 
# Returns (rotation_deg or -rotation_deg), new_variance, rotated_vectors

def bestRotationOnAxes(optimisation_axis, dimension, axis1, axis2, rotation_deg, vectors):
    # Generate orthogonal vectors describing the plane
    plane_v1, plane_v2 = planeToVectors(dimension, axis1, axis2)
    
    # Try rotating left (anti-clockwise)
    vectors_left = rotateOnPlane(plane_v1, plane_v2, -rotation_deg, vectors)
    points = unzip(vectors_left)[optimisation_axis]
    variance_left = stats.variance(points)

    # Try rotating right (clockwise)
    vectors_right = rotateOnPlane(plane_v1, plane_v2, rotation_deg, vectors)
    points = unzip(vectors_right)[optimisation_axis]
    variance_right = stats.variance(points)

    # Choose rotation the maximises the variance
    if variance_left > variance_right:
        return -rotation_deg, variance_left, vectors_left
    else:
        return rotation_deg, variance_right, vectors_right
```


```
# Rotates the points by [rotation_deg]° on every plane allowed to improve the variance along the optimisation_axis
# 
# To improve the X-axis alignment, the points are allowed to rotate on every plane,
# but to improve the Y-axis, the rotations may not affect the X-axis alignment
# In 3D: For X-axis alignment rotation on XY and XZ plane (YZ is also allowed, but will not affect X-axis)
#        For Y-axis alignment rotation on YZ, but not YX plane

def improveAxisAlignmentBy(optimisation_axis, dimensions, rotation_deg, vectors):
    if optimisation_axis == dimensions-1:
        points = unzip(vectors)[optimisation_axis]
        return stats.variance(points), vectors
        # TODO

    # Axes to pair with optimisation_axis to form the plane to rotate on.
    # Assumes all axes < optimisation_axis to be fixed and immutable
    for axis2 in range(optimisation_axis+1, dimensions):
        angle_diff, var, vectors = bestRotationOnAxes(
            optimisation_axis = optimisation_axis,
            dimension = dimensions,
            axis1 = optimisation_axis,
            axis2 = axis2,
            rotation_deg = rotation_deg,
            vectors = vectors
        )

    return var, vectors
```


```
# Improve alignment along the optimisation_axis by rotating on the given plane (axis1, axis2)
# until a maxima in the variance is reached.
def optimiseAxisOnPlane(optimisation_axis, dimensions, axis1, axis2, rotation_deg, vectors):
    var_old = -1
    var = 0

    while var > var_old:
        var_old = var
        angle_diff, var, vectors = bestRotationOnAxes(
                optimisation_axis = optimisation_axis,
                dimension = dimensions,
                axis1 = axis1,
                axis2 = axis2,
                rotation_deg = rotation_deg,
                vectors = vectors
            )

    return var, vectors
```

The following helper function `approximatePC` approximates the optimal rotation to maximise the variance along the given axis. However, instead of starting at 0° and incrementally rotating by 1° until the optimum is reached, we can try an approach similar to binary search, where we "search" for the best rotation by initially making big jumps and then refining our search space: We start by rotating 90°, then 45°, then 22.5°, etc. until we have reached the desired precision (e.g. ≤1°). To reach a precision of 1° this method would take up to 8 steps instead of up to 90 when incrementing by 1°. If the initial efficiency of simply incrementing was $\Theta(n)$, the new algorithm runs in $\Theta(\lg n)$.

Since the relationship between rotation and variance is not always linear, I found that a factor of 1.5 produces better results than the factor of 2 mentioned above. Hence, we would be rotating by 90°, then 60°, 40°, 26.7°, etc. Finding the best angle would now take 13 steps instead of 8, but we still have an efficiency of $\Theta(\log_{1.5} n) = \Theta(\lg n)$ with a more accurate outcome. By decreasing the factor we can make it still more accurate but it would need longer to run.


```
# Find the nth principal component (target_pc) by optimising variance along its corresponding axis
# by rotating on all available planes.
# Makes sure to not affect previous principal components.
#
# It is highly recommended to execute it in order of increasing PC number

def approximatePC(from_dimension, target_pc, precision, vectors, log_factor=1.5):
    # Requires that prior PCs are already approximated in the input vectors

    var_old = -1
    var = 0

    # Instead of improving by a little bit every time, initially make big adjustments and get finer and finer
    # Approach is inspired by binary search.
    angle = 90  # Start at 90° changes instead of 180°, since flipping by 180° results in the same variance
    while angle > precision:
        var_old = var
        var, vectors = improveAxisAlignmentBy(target_pc, from_dimension, angle, vectors)

        angle = angle / log_factor


    return var, vectors
```


```
# Simulates PCA for
#  any number of input dimensions >= 2
#  and any number of desired Principal Components <= no. of dimensions
def myPCA(input_vectors, components, log_factor=1.5):
    precision = .1  # Smallest rotation-angle size (in degrees)

    dimensions = len(input_vectors[0])
    target_dimensions = components  # No. of Principal Components to find

    if target_dimensions > dimensions:
        raise Exception(f'Requested more components ({components}) than input data has dimensions ({dimensions})')

    variances = np.zeros(target_dimensions)

    vectors_new = np.copy(input_vectors)

    # Compute Principal Components
    for pc in range(target_dimensions):
            variances[pc], vectors_new = approximatePC(dimensions, pc, precision, vectors_new, log_factor)

    # Calculate variances explained by components
    total_var = 0
    for points in unzip(input_vectors):
        total_var += stats.variance(points)
    varsExplained = [var / total_var for var in variances]

    return {
        "variance explained": varsExplained,
        "components": unzip(vectors_new)[:target_dimensions]
    }
```

#### Demsonstration of approximation algorithm `myPCA`


```
vectors3d = transpose(genNdData(n=3, size=100))
results = myPCA(vectors3d, 3)
results['variance explained']
```




    [0.8108474450655777, 0.1233672858781378, 0.06578526905628443]




```
vectors5d = transpose(genNdData(n=5, size=100))
results = myPCA(vectors5d, 3)
results['variance explained']
```




    [0.5593309564391039, 0.1588819208072347, 0.12838213614982016]




```
vectors10d = transpose(genNdData(n=10, size=100))
results = myPCA(vectors10d, 3)
results['variance explained']
```




    [0.428243962604282, 0.10745071173485049, 0.10147638091951613]




```
df = pd.DataFrame(transpose(results['components']))
sns.pairplot(df)
plot3d(transpose(results['components']))
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>                <div id="1341fa94-31f7-4fd0-9f19-ba1538346c3f" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("1341fa94-31f7-4fd0-9f19-ba1538346c3f")) {                    Plotly.newPlot(                        "1341fa94-31f7-4fd0-9f19-ba1538346c3f",                        [{"hovertemplate": "x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "", "scene": "scene", "showlegend": false, "type": "scatter3d", "x": [-0.04663827910493279, -0.10680069468582146, 0.35739067211141734, -0.10665627057286026, 0.13500936502878766, 0.1975447849536358, 0.3894313518945944, -0.04784822687573527, -0.040967562896971454, 0.07262906368073803, 0.25078078619679023, 0.10259360647323695, 0.12376315692453943, 0.03557072244938227, 0.06519653586646401, -0.072788895900261, 0.030119973621673596, 0.30859401841260986, -0.03216524851206539, -0.010614419022013927, 0.6641644934989237, -0.03231046288418097, 0.05260456570352015, 0.5730428609426718, 0.12687253615163838, -0.016680523704927577, -0.03611295342686414, 0.07108640506909476, -0.13260691289318627, -0.08349127953441407, -0.07428426931046027, 0.02182989239686334, 0.10223580703404146, 0.01927240179154273, 0.10687251925900926, 0.12080971391275791, 0.15993988168138318, 0.05557453350822207, 0.531910642211521, 0.623019135977584, 0.17695998023030685, -0.0613489312396901, -0.0719618791804263, 0.38153124593263915, 0.013372248656529178, 0.025711052332935327, 0.10731100844906015, 0.007149724821324425, 0.07366238487996751, 0.4931135449160775, 0.20893499516991854, 0.16717798706311038, -0.020439915626612275, 0.1026875604419305, 0.061557386593618965, -0.0037849092635092382, -0.036749503852634265, -0.004471979639748846, 0.1682071160354069, 0.02111794149032221, 0.508608468538478, 0.04622237196724751, 0.051129617015730225, 0.024995316354823504, -0.021106019497150887, -0.08442506777338153, -0.09947360360437224, -0.03533397687731892, -0.12123097314304532, 0.10553225687525468, -0.03090834535835771, -0.008323938000140023, 0.10825471232731831, 0.6938888804266965, -0.1191503615413145, 0.06433515728364869, 0.08249937333424978, -0.08961767127649244, -0.09285330849128198, 0.08133063605298398, -0.12107652326088938, 0.1910390744750052, -0.01847208523083569, -0.059529058596612706, 0.027002852875844207, -0.05254966643848825, -0.07485610356188944, 0.019625144940665314, 0.26610802236320014, 0.10488405254738543, 0.2560326825140845, 0.23891040481830308, 0.0486623522706634, -0.09368931780924569, -0.04291490380693294, 0.7642333885751899, 0.12477009570046826, 0.1511102543083845, 0.03425936886704412, 0.1752365205811549], "y": [-0.2806444169406752, 0.022529871580042916, -0.07256796230852061, 0.08931511300054526, -0.1511052523146782, -0.019618448361271183, 0.1283394210149132, -0.0024926419965940485, -0.07895659822077515, -0.05126339816097871, 0.11498260328901914, 0.009015773545198074, 0.06954849532151772, -0.023663243311831134, 0.08767657338297026, -0.03282337425495359, 0.10697251737029194, -0.06851730328891764, -0.04632031372457079, -0.02871401887015916, 0.12327534463552135, -0.022950867948988683, -0.1600600754992261, -0.10641815100379336, -0.07072680402901511, -0.05943151498180994, 0.08955379923946956, -0.08304660341486045, -0.027817104910156362, 0.15299403465723568, 0.056755288898456116, 0.07956067076365614, 0.14379961158203763, -0.0018836084753959621, -0.101099957390821, 0.027098434863772353, 0.18687222936921477, -0.01670474402299129, -0.019137122783690386, -0.10363351932785385, -0.06295291605760192, 0.07923464038539658, 0.04948811022022892, -0.002833608532120357, 0.21637299573371382, 0.08556471873577347, -0.04022790095083183, -0.03729208030043847, -0.005846302374465777, 0.15348299214999714, -0.11072708496580265, 0.011792040260801353, -0.16289405679781382, 0.0388811618172711, 0.010635475946864321, -0.10680235170273035, 0.19911956640993425, -0.0035667239587364686, -0.05669194239929324, 0.0461642642266431, -0.0444713947082169, -0.014564134267689452, -0.061987313603937465, 0.05824508986911111, 0.12706844660097544, -0.009685781127535655, -0.018145281327146788, -0.024858711616198945, -0.09237690491319633, -0.01931250760089069, 0.05150299569933688, -0.22345131612419017, -0.09114935614712305, 0.02021283649064086, -0.02980691310204479, -0.25824434266994095, -0.15412386763067665, -0.13588393242545543, 0.09688568396539461, -0.05484474790571547, 0.1321149161995024, -0.11270209365250893, 0.08122128881733706, -0.09728248399735508, -0.05974522274045262, 0.080744584474942, -0.0580151344671855, -0.1849580950906106, 0.018280564078965632, -0.04995346041495003, -0.005922612859605291, -0.05363711835050341, -0.12778779006686786, 0.02717940021800208, -0.03963148719686503, -0.021078691039208487, -0.11047077696289212, -0.013536249759051861, -0.03753748486906691, 0.08662288205126445], "z": [0.06862865113420537, 0.12383636440870509, 0.0968059149166938, 0.0683443490482144, 0.0396883856270892, 0.23059722867983187, 0.060435000992947914, 0.023207277126690393, 0.08176870318440507, 0.1440302547383876, 0.21887629650605145, 0.03551603697678453, 0.047536056401230924, 0.048652110635187255, 0.14924801094904494, 0.043040254592363124, -0.07323406819984214, 0.06215245739247002, -0.12466017148448207, 0.02210270223632946, 0.02251187085095366, -0.0200967674940772, -0.016355585561192012, 0.10916343936415361, 0.1645899059615793, -0.013109942202362656, 0.04040082781867625, -0.022087076889549126, 0.03497585789029709, -0.02395281235973624, -0.01788559639375055, 0.06297118757000086, -0.12143499787116688, 0.18943168618713718, -0.13023543507482804, 0.23134608211310168, 0.08559776162171819, 0.20037325561815433, -0.1278749831654809, -0.07590360448246448, 0.10266786048783981, 0.08654096052967569, 0.11481778882629588, 0.1255191364139529, -0.08794211861580764, -0.0027741396138650882, -0.05553047487573155, -0.000391799786113532, -0.062051757656702455, -0.062317852882466135, 0.17225726932678484, 0.11090318469004526, 0.1216790839235288, 0.11903099659714232, 0.006575198668493327, -0.021953342988455398, 0.050591350107511576, 0.020358029953553992, 0.15473504266625973, -0.02043362939430373, 0.04722334361302469, 0.07641816180417839, -0.03340579208943318, 0.010839215883246916, 0.03068624131070915, -0.143960204304404, 0.217135023272737, 0.0007753927477499432, 0.04296503884321037, 0.13308884481003644, 0.011698387605634675, -0.14620684879031737, 0.01370012061064447, 0.1069082078220401, -0.09703494455386132, 0.0769702357339479, -0.09772900939152826, -0.0462556711118172, 0.0254736794504348, 0.10414475493191691, -0.09605338416473472, -0.06112700711800715, 0.11299537950137509, -0.004440290706192378, 0.19601900939629072, 0.02923539080163347, 0.06806663411578667, -0.00875272153758399, 0.18721599939210634, 0.06404721742928592, -0.1230555438397371, -0.02057082504619617, 0.09690855133084143, 0.04548792763915996, 0.12459670978469732, -0.13211194072255694, 0.209759806336821, 0.0554894985018575, 0.042059644934589475, 0.047146167799252986]}],                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "scene": {"domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "xaxis": {"title": {"text": "x"}}, "yaxis": {"title": {"text": "y"}}, "zaxis": {"title": {"text": "z"}}}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('1341fa94-31f7-4fd0-9f19-ba1538346c3f');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



    
![png](notebook_files/notebook_44_1.png)
    


#### Comparison with standard PCA


```
# PCA
X = np.ndarray.transpose(np.array(unzip(vectors10d)))

pca = decomposition.PCA(n_components=3)
X_reduced = pca.fit_transform(X)
print("Standard PCA explained variances:", pca.explained_variance_ratio_)

myPCAErrors = np.absolute(results['variance explained'] - pca.explained_variance_ratio_) / pca.explained_variance_ratio_
print("myPCA errors per component:", myPCAErrors)
```

    Standard PCA explained variances: [0.42824768 0.10745068 0.10149645]
    myPCA errors per component: [0.00000868 0.00000033 0.00019773]


# Conclusion

While it was clear from the beginning that I wasn't going to discover a revolutionary new approach to computing PCAs that is much faster than the tried and tested approach that many have worked on over the years, this approach is arguably much more intuitive to understand and is a very reasonable approximation of the standard approach (typical errors of < 0.0001% for each component).

I have learnt a good amount about both, PCA as well as rotations in n-dimensional space while researching for this project and hope that you, whoever may end up reading this, have too.

<a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=3c882279-82a6-4796-9acf-70d58cb2aa76' target="_blank">
<img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
