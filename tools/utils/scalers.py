def min_max_scale(x, min, max):
    if x < min:
        return 0
    elif x > max:
        return 1
    else:
        return (x - min) / (max - min)