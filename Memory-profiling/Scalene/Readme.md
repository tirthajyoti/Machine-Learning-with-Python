## [Scalene](https://github.com/plasma-umass/scalene)

### Install
`pip install scalene`

### Run (on CLI)
`$ scalene <yourapp.py>`

### Example output

If you run the `mlp.py`

`$ scalene mlp.py`

You may see something like,

![scalene-1](https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Memory-profiling/Scalene/scalene-1.PNG)

### Features
Here are some of the cool features of Scalene. Most of them are self-explanatory and can be gauged from the screenshot above,

- **Lines or functions**: Reports information both for entire functions and for every independent code line
- **Threads**: It supports Python threads.
- **Multiprocessing**: supports use of the multiprocessing library
- **Python vs. C time**: Scalene breaks out time spent in Python vs. native code (e.g., libraries)
- **System time**: It distinguishes system time (e.g., sleeping or performing I/O operations)
- **GPU**: It also can report the time spent on an NVIDIA GPU (if present)
- **Copy volume**: It reports MBs of data being copied per second 
- **Detects leaks**: Scalene can automatically pinpoint lines responsible for likely memory leaks!
