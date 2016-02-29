# A reduction function receive a Data-Set - x,
# and returns a number to reduce data-set to.
# Add reduction functions here in order to dictate the reduction pattern and reference it from your file.


# basic reduction function - halves the data size
def half_reduction_function(x):
    return len(x[0]) / 2


# a function which returns a reduction function which reduces i features in each iteration
def minus_i_reduction_function(i):
    def func(x):
        return len(x[0]) - i
    return func


# -1 reduction function
def minus_one_reduce_function(x):
    return (minus_i_reduction_function(1))(x)


# A function which returns a precise reduction function which halves the data size until it is smaller than n and then
# it descends gradually.
def precise_reduce_function(n):
    def func(x):
        for i in range(0, 10):
            if (len(x[0]) / (2 - (i / 10))) > n:
                return len(x[0]) / (2 - (i / 10))
        return len(x[0]) - 1
    return func


# Reduction function that reduce more for bigger data and less for smaller
def gradient_reduction_function(x):
    if len(x[0]) > 10000:
        return 10000
    if len(x[0]) > 1000:
        return len(x[0]) - 1000
    if len(x[0]) > 100:
        return len(x[0]) - 100
    if len(x[0]) > 10:
        return len(x[0]) - 10
    else:
        return len(x[0]) - 1


# Reduction function as described in "deepsign" article
def deep_sign_reduction_function(x):
    if len(x[0]) > 5000:
        return 5000
    if len(x[0]) > 2500:
        return 2500
    if len(x[0]) > 1000:
        return 1000
    if len(x[0]) > 500:
        return 500
    if len(x[0]) > 250:
        return 250
    if len(x[0]) > 100:
        return 100
    if len(x[0]) > 30:
        return 30
    return len(x[0]) / 2


