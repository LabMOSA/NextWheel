# NextWheel Python module

This is the Python module for the [NextWheel project](https://github.com/LabMOSA/NextWheel) - Open design of next-generation instrumented wheelchair wheels

To install:

```
pip install nextwheel
```

To upgrade:

```
pip install --upgrade nextwheel
```

To use the module:

```python
>>> from nextwheel import NextWheel

>>> nw = NextWheel("xx.xx.xx.xx")
>>> nw.start_streaming()

>>> print(nw.fetch())
    
>>> nw.stop_streaming()
```

Streamed data is sent by the wheel over UDP (by default, broadcast on port
12345). Make sure your firewall accepts incoming UDP datagrams on this port.
To change where the wheel sends its data (e.g., a different port for each
wheel, or this computer's IP address instead of broadcast):

```python
>>> nw.start_streaming("255.255.255.255", 12346)
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
