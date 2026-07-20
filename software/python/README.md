# NextWheel Python module

To install the Python module:

```
pip install git+https://github.com/felixchenier/NextWheel.git#subdirectory=software/python
```

To upgrade the Python module:

```
pip install --upgrade git+https://github.com/felixchenier/NextWheel.git#subdirectory=software/python
```

To use the module:

```python
>>> from nextwheel import NextWheel

>>> nw = NextWheel("xx.xx.xx.xx")
>>> nw.start_streaming()

>>> print(nw.fetch())
    
>>> nw.stop_streaming()
```

To see the list of recorded data files:

```python
>>> nw.file_list()
```

To download a file:

```python
>>> nw.file_download(filename)
```

To erase a file:

```python
>>> nw.file_delete(filename)
```

To read a file:

```python
>>> data = NextWheel.read_dat(filename)
```
