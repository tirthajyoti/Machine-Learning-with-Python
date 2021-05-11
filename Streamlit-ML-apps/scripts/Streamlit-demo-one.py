import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title('Streamlit app demo (with Numpy, Pandas, Scikit-learn)')
"""
## Dr. Tirthajyoti Sarkar, Fremont, CA, July 2020
[My LinkedIn profile](https://www.linkedin.com/in/tirthajyoti-sarkar-2127aa7/),
[My Github profile](https://github.com/tirthajyoti.)

---
### What are we covering in this app/demo?

In this demo, we will cover the following aspects with Streamlit,

- Basic markdown
- Displaying image
- LaTeX and code rendering
- Python objects rendering
- Numpy arrays rendering
- Working with Pandas DataFrame
    - Filtering data
    - Bar chart
    - Line chart (Altair)
- Widget magic and interactivity
- Pyplot (Matplotlib graphs)
- A linear regression problem (interactive)

### What's the goal?
The primary goal in this app is to show the application of Streamlit working
synergistically with **Python objects** - numbers, strings, Numpy arrays,
Pandas DataFrames, Matplotlib graphs, Scikit-learn estimators,
and **interactive web-app elements** - textboxes, sliders, file-explorer, etc.

As a secondary goal, we illustrate the **rendering capabilities** of Streamlit for
other type of content such as LaTeX, code, markdown, images, etc.

### How to run this app?
We basically write a Python script called `Streamlit-demo-one.py` and
just run it with the following command on the terminal,

```streamlit run Streamlit-demo-one.py```

It starts a server and we point to `localhost:8501` to see this app.
"""

"""
---
## Some basic markdown
We start off by showing some basic markdown syntaxt rendering.
Streamlit can handle markdown content just like your Jupyter notebook.

We just need to put the markdown text within two pairs of multiline comment
symbols like
`""\" {markdown text here...} ""\"`.

Here is a **bold** text in *italic* format.

Here is a $$L^AT_EX$$ equation:
$$E(m_0,v)=\\frac{m_0.c^2}{\sqrt{1-\\frac{v^2}{c^2}}}$$.

And here is my [home page](https://tirthajyoti.github.io) i.e. **Github.io**.
"""

"""
---
## Displaying image

The default markdown image tag is not suitable for controlling the image size.
So, we should use `st.image()` method to display image.

Here is a screenshot from the Streamlit website.
The image is hosted on my
[Github repo](https://github.com/tirthajyoti/Machine-Learning-with-Python)
and we just pass on the URL.
"""
st.code('''
image_URL = "https://raw.githubusercontent.com/tirthajyoti/
Machine-Learning-with-Python/master/Images/st-1.PNG"

st.image(image_URL, width=800)
''')

st.image("https://raw.githubusercontent.com/tirthajyoti/\
Machine-Learning-with-Python/master/Images/st-1.PNG",
    width=800)

"""
---
## Special function for LaTeX rendering

We can separately use `st.latex()` to render latex content.

```
st.latex(r'''
a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
\sum_{k=0}^{n-1} ar^k =
a \left(\frac{1-r^{n}}{1-r}\right)
''')
"""

st.latex(r'''
a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
\sum_{k=0}^{n-1} ar^k =
a \left(\frac{1-r^{n}}{1-r}\right)
''')

"""
---
## Code rendering
We can use `st.code()` to render code blocks nicely with optional
syntax highlighting.

```
code_block = '''a, b = 10, 20
def add(x,y):
        return x+y
print(add(a,b))'''

st.code(code_block,'python')
"""

"""This results in the following..."""

code_py = '''a, b = 10, 20
def add(x,y):
        return x+y
print(add(a,b))'''

st.code(code_py,'python')

"""
Some JavaScript code,

```
code_js = '''
let a = 10;
const b = parseFloat('20.5');
add = (x,y) => x + y
console.log(add(a,b))
'''
st.code(code_js,'javascript')
"""

"""Results in..."""

code_js = '''
let a = 10;
const b = parseFloat('20.5');
add = (x,y) => x + y
console.log(add(a,b))
'''

st.code(code_js,'javascript')

"""
---
## Native Python objects are rendered pretty

Python objects like list and dictionaries are rendered in a
pretty and visually appealing manner. We use the versatile `st.write()` method
for all such rendering.
"""

"""
### A list
Here is how the list `list1 = ['Apple', 2.5, [-3,3]]` gets renderd...
"""
list1 = ['Apple', 2.5, [-3, 3]]
st.write("list1: ", list1)

"""
### A tuple
Here is how the list `tuple1 = ((1,2),(100,110),(35,45))` gets renderd...
"""
tuple1 = ((1, 2), (100, 110), (35, 45))
st.write("tuple1: ",tuple1)

"""
### A dictionary
Here is how the dict `dict1 = {'Item1':10, 'Item2':[1,3,5],'Item3':-5}` gets rendered...
"""
dict1 = {'Item1':10, 'Item2':[1,3,5],'Item3':-5}
st.write("dict1: ", dict1)

"""
### A function
The docstring/description of the function is rendered by the `st.write()` method.
For example, we define,

```
def square (x):
    \"""
    Squares a given number
    \"""
    return x*x
"""

def square (x):
    """
    Squares a given number
    """
    return x*x

st.write(square)

"""
---
## Numpy arrays

Numpy arrays (one- and two-dimensional) are also rendered nicely
by the `st.write()` method,
although for long arrays the vertical rendering can become unwieldy.

```
a = np.arange(1, 20, 2) #Positive odd integers up to 20
st.write(a)
"""

a = np.arange(1, 20, 2)
st.write(a)

"""
### Two-dimensional arrays
```
b = np.arange(1, 21).reshape(5, 4)
st.write(b)
"""
b = np.arange(1, 21).reshape(5, 4)
st.write(b)

"""
### Three-dimensional arrays (rendered normally)
```
c = np.arange(1, 21).reshape(5, 2, 2)
st.write(c)
"""
c = np.arange(1, 21).reshape(5, 2, 2)
st.write(c)

"""
### The transpose

```
st.write(b.T)
"""
st.write(b.T)

"""
---
## Working with Pandas DataFrame

We can render a Pandas DataFrame either by using `st.write()` or `st.dataframe()`
methods.

"""
code_df = '''
# Random data-filled coulmns
df = pd.DataFrame(np.random.normal(loc=5,
scale=5, size=50).reshape(10, 5),
columns = ['A'+ str(i) for i in range(1, 6)])

# Two derived columns
df['A6'] = 10*np.sin(df['A1'])
df['A7'] = 0.1*df['A2']**2

st.write(df)
'''

st.code(code_df)

"""
### Page refresh generates new data
Every time the page refreshes, the code generates new random data,
and the plot below regenerates as well.
"""
# Random data-filled coulmns
df = pd.DataFrame(np.random.normal(loc=5,
scale=5, size=50).reshape(10, 5),
columns = ['A'+ str(i) for i in range(1, 6)])

# Two derived columns
df['A6'] = 10*np.sin(df['A1'])
df['A7'] = 0.1*df['A2']**2

st.write(df)

"""
### Applying a filter on the DataFrame

We filter the DataFrame by selecting only those rows where `A1` > 0 and `A3` > 3.
Note that due to the random nature of the DataFrame generation, **there is no guarantee that
we will get a non-empty DataFrame every time we re-run the code**.
"""

code_df2 = '''
df_filtered = df[(df['A1']>0) & (df['A2']>3)]
st.write(df_filtered)'''

st.code(code_df2)

df_filtered = df[(df['A1']>0) & (df['A2']>3)]
st.write(df_filtered)

"""
### Now, write the filtered DataFrame on the disk
We can easily ask the user a filename and write the filtered data to that file!
"""

csv_filename = str(st.text_input("Enter a filename for saving the DataFrame as a CSV file",
                                max_chars=30))

if ('.csv' not in csv_filename and len(csv_filename)>0):
    csv_filename += ".csv"
if len(csv_filename)>0:
    df_filtered.to_csv(csv_filename)
    st.markdown("#### File was saved.")
else:
    st.markdown("#### No filename was provided. Nothing was saved.")

"""
### Reading a CSV from the web

Reading data from a remotely hosted file (and rendering in a DataFrame)
is as easy as the short code below,
"""
code_df_csv = '''
data_url = "https://raw.githubusercontent.com/tirthajyoti/
D3.js-examples/master/html/data/worldcup.csv"
df_csv = pd.read_csv(data_url)
df_csv=df_csv.shift(2,axis=1).reset_index().drop(['team','region'],axis=1)
df_csv.columns = ['team','region','win','loss','draw','points','gf','ga','cs','yc','rc']
st.write(df_csv)
'''

st.code(code_df_csv)

data_url = "https://raw.githubusercontent.com/tirthajyoti/D3.js-examples/master/html/data/worldcup.csv"
df_csv = pd.read_csv(data_url)
df_csv=df_csv.shift(2,axis=1).reset_index().drop(['team','region'],axis=1)
df_csv.columns = ['team','region','win','loss','draw','points','gf','ga','cs','yc','rc']
st.write(df_csv)

"""
### A simple bar chart using Pandas built-in `plot` module
"""
code_bar = '''
# Goal difference => gf - ga
df_csv['gd'] = df_csv['gf'] - df_csv['ga']
fig=df_csv.sort_values(by='gd', ascending=False)[['team','gd']].plot.bar(x='team',
                                                y='gd',figsize=(7, 6))
plt.grid(True)
plt.title("Goal difference bar chart")
plt.xticks(rotation=30)
st.pyplot()
'''

st.code(code_bar)

# Goal difference => gf - ga
df_csv['gd'] = df_csv['gf'] - df_csv['ga']
fig=df_csv.sort_values(by='gd', ascending=False)[['team','gd']].plot.bar(x='team',
                                                y='gd',figsize=(7, 6))
plt.grid(True)
plt.title("Goal difference bar chart")
plt.xticks(rotation=30)
st.pyplot()

"""
## Line chart with Altair library

We take some of the columns from the DataFrame and create a line chart.
This line chart is based on the
[`Altair` library](https://altair-viz.github.io/getting_started/overview.html)
charting function.
You can zoom and pan the chart and even see the HTML code behind it.
"""

st.line_chart(df[['A1', 'A2', 'A6', 'A7']])

"""
---
## Widget magic

Below we are showing the evaluation of the
function $$f(x)=\sin(x).e^{-0.1x}$$ with the help of a simple slidebar widget.
```
def f(x):
    return np.sin(x)*np.exp(-0.1*x)
"""

def f(x):
    return np.sin(x)*np.exp(-0.1*x)


"""
The slidebar widget is created by this code,
```
x = st.slider('x', -8, 8)
"""
x = st.slider('x', -8, 8)

"""
### Function value
The variable `x` is defined above as the returned value from the slidebar widget.
Therefore, we can dynamically evaluate the `f(x)` by passing on this `x` value
as we move the slider up and down.

We are printing the function value below. Move the slidebar and see how the
evaluation changes.
"""
st.write(f"$f(x)$ evaluated at {x} is: "+str(round(f(x), 3)))

"""
---
## A Matplotlib graph of the function
The code below graphs the function above using plain vanila `Matplotlib` and
a single `Streamlit` call `st.pyplot()` for rendering.

This chart, unlike the Altair chart above, is **not a dynamic chart**.
However, note that the `Matplotlib` code contains fair bit of sophistication
(even a LaTeX formatted string in the title). All of that is flawlessly handeled
by the `st.pyplot()` function.
"""
code_plt = '''
# Some plain vanila Matplotlib code
var_x = np.arange(-8, 8, 0.2)
var_y = np.apply_along_axis(f, 0, var_x)
plt.figure(figsize=(7,4))
plt.title("Plot of $sin(x).e^{-0.1x}$",
fontsize=16)
plt.scatter(var_x, var_y,
c='green', alpha=0.5)
plt.plot(var_x, var_y,
c='k')
plt.grid(True)

#This is the Streamlit callback
st.pyplot()
'''

st.code(code_plt, 'Python')

# Some plain vanila Matplotlib code
var_x = np.arange(-8, 8, 0.2)
var_y = np.apply_along_axis(f, 0, var_x)
plt.figure(figsize=(7, 4))
plt.title("Plot of $sin(x).e^{-0.1x}$", fontsize=16)
plt.scatter(var_x, var_y, c='green', alpha=0.5)
plt.plot(var_x, var_y, c='k')
plt.grid(True)

# This is the Streamlit callback
st.pyplot()

"""
---
## A (simple) linear regression problem

Next, we show, how to generate a linear regression dataset with **tunable level
of noise** using simple widgets from Streamlit. In the previous two sections,
we introduced the slider widget and the pyplot. In this section, we combine them
in a dynamic fashion.

### One-dimensional linear regression problem

A simple linear regression (with one variable) can be written as,
"""
st.latex(r'''y = a_1.x+ b_1+ N(\mu, \sigma^2)''')

"""
where,

$y$ : The observed data,

$x$ : The feature data,

$N(\mu, \sigma^2)$ : The **noise drawn from a Gaussian Normal distribution**

$\mu$ : The mean of the noise

$\sigma^2$ : The variance of the noise
"""
"""
### Adjust the noise to dynamically generate a linear regression dataset

We choose $a_1=2.5$ and $b_1=5$ for the illustration.

Below, the sliders can be adjusted to tune the level of the noise.
**Every time you move the sliders, you essentially generate a new
linear regression problem**
(with the same features but with slightly different observed data).

The data vs. feature plot, which is dynamically updated, is shown below
to illustrate this point.

Move the "Noise std. dev" slider all the way from left end to right end to
observe the impact on the observed data. **Do you see the observed data becoming
_more noisy_ as you mode the slider towards right**?
"""

feature = np.arange(1,10,0.1)
noise_mu = st.slider("Noise mean",
                    min_value=-4.0,
                    max_value=4.0,
                    value=0.0,
                    step=0.5)
noise_std = st.slider("Noise std. dev",
                    min_value=0.0,
                    max_value=3.0,
                    value=1.0,
                    step=0.1)
len_dataset = feature.size
data = 2.5*feature+5+np.random.normal(noise_mu,
                                                noise_std,
                                                size=len_dataset)
def plot_xy(x,y):
    plt.figure(figsize=(5, 4))
    plt.title("Data vs. feature", fontsize=12)
    plt.scatter(x, y, c='orange', edgecolor='k', alpha=0.5)
    plt.xlabel("Feature values", fontsize=10)
    plt.ylabel("Observed data (with noise)", fontsize=10)
    plt.xlim(0,12)
    plt.ylim(0,35)
    #plt.plot(x, y, c='k')
    plt.grid(True)

fig = plot_xy(feature, data)
st.pyplot(fig)

"""
### Fitting a model with the data (using `scikit-learn`)

Next, we fit a `LinearRegression()` model from the famous `scikit-learn` package
with our data, and show the model coefficients and the $R^2$ metric.

Note, how they change slightly with each iteration of a new problem generation.
Note that we chose $a_1=2.5$ and $b_1=5$ and the estimates should come close to
these numbers.

**Move the sliders (above) around and observe the changes in the estimates of the
linear coefficient and the bias term**. Note that the noise mean primarily impacts
the bias term whereas noise std. dev primarily impacts the linear coefficient.
You will also notice that the $R^2$ score generally becomes lower as the
noise std. dev increases i.e. **the linear model has a hard time explaining the
variance in the observed data (if the spread of the noise is high)**.
"""

code_sklearn='''
from sklearn.linear_model import LinearRegression

data = data.reshape(-1, 1)
feature = feature.reshape(-1, 1)

lr = LinearRegression()
lr.fit(feature, data)

lr_coef = round(float(lr.coef_[0]), 3)
lr_bias = round(float(lr.intercept_), 3)
lr_r2 = round(float(lr.score(feature, data)), 3)

st.write("The linear coefficient estimated: ", lr_coef)
st.write("The bias term estimated: ", lr_bias)
st.write("The R^2 score estimated: ", lr_r2)
'''

st.code(code_sklearn)

data = data.reshape(-1, 1)
feature = feature.reshape(-1, 1)
lr = LinearRegression()
lr.fit(feature, data)

lr_coef = round(float(lr.coef_[0]), 3)
lr_bias = round(float(lr.intercept_), 3)
lr_r2 = round(float(lr.score(feature, data)), 3)

st.write("The linear coefficient estimated: ", lr_coef)
st.write("The bias term estimated: ", lr_bias)
st.write("The R^2 score estimated: ", lr_r2)
