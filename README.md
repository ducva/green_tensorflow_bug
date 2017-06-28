# create conda env
`conda env update`

`source activate test_green`

# Run test with green:
`green test.py`

```,output
$ green test.py
Before load graph
Before create session
After creating session

<STUCK HERE>

```

# Run test with python

`python test.py`
```, output
$ python test.py
Before load graph
Before create session
After creating session
Test data: [[ 0.   0.5  0.   0.   0.   0.   0.   0.   2.   0.   0.   0.   0. ]]
Prepare to run Tensorflow Session
Finish run Tensorflow session
.
----------------------------------------------------------------------
Ran 1 test in 0.012s

OK
```
