import numpy as np


def deg_to_rad(deg):
    return deg * np.pi / 180


def get_1st_layer_degs():
    list_of_degs = []
    lg = 30
    deg = 90 + (lg / 2)
    for i in range(12):
        list_of_degs.append(deg)
        deg += lg
    return list_of_degs


def get_2nd_layer_degs():
    list_of_degs = []
    sm = 6
    lg = 12
    deg = 90 + (lg / 2)
    for j in range(12):
        list_of_degs.append(deg)
        for i in range(3):
            deg += sm
            list_of_degs.append(deg)
        deg += lg
    return list_of_degs


def get_3rd_layer_degs():
    list_of_degs = []
    sm = 1.5
    lg = 3.0
    deg = 90 + (lg / 2)
    for i in range(16 * 3):
        list_of_degs.append(deg)
        for i in range(3):
            deg += sm
            list_of_degs.append(deg)
        deg += lg
    return list_of_degs


def get_index(lst1, lst2, lst3, pick="BWB"):
    idx_1st = []
    for idx, c in enumerate(lst1):
        if c == pick[0]:
            idx_1st.append(idx)
    idx_2nd = []
    for idx, c in enumerate(lst2):
        prev_idx = idx // 4
        if (prev_idx in idx_1st) and c == pick[1]:
            idx_2nd.append(idx)
    idx_3rd = []
    for idx, c in enumerate(lst3):
        prev_idx = idx // 4
        if (prev_idx in idx_2nd) and c == pick[2]:
            idx_3rd.append(idx)
    return idx_1st, idx_2nd, idx_3rd


dim = (580, 580)
origin = (dim[0] / 2, dim[1] / 2)
alpha = 0.3
sw = 2

svg_str = '<svg width="{}" height="{}">{}</svg>'
circle = '<circle cx="{}" cy="{}" ' +\
    'r="{}" stroke="black" stroke-width="1.5" fill="{}" ' +\
    'stroke-opacity="{}" fill-opacity="{}" />'
line = '<line x1="{}" y1="{}" x2="{}" y2="{}" ' +\
    'stroke="black" stroke-width="{}" stroke-opacity="{}" />'

clr_dict = {"B": "blue", "W": "white"}

list_of_1st_layer_degs = get_1st_layer_degs()
list_of_2nd_layer_degs = get_2nd_layer_degs()
list_of_3rd_layer_degs = get_3rd_layer_degs()
list_of_1st_layer_colors = list("BWWWBBWWBBBW")
list_of_2nd_layer_colors = list("BWWW" * 4 + "BBWW" * 4 + "BBBW" * 4)
list_of_3rd_layer_colors = list("BWWW" * 16 + "BBWW" * 16 + "BBBW" * 16)
indexes = get_index(list_of_1st_layer_colors,
                    list_of_2nd_layer_colors,
                    list_of_3rd_layer_colors)


def circle_f(origin, radius, degree):
    x = np.cos(deg_to_rad(degree)) * radius
    y = np.sin(deg_to_rad(degree)) * radius
    return (np.round(x + origin[0], 2), np.round(origin[1] - y, 2))


def draw_origin_cirle():
    return circle.format(origin[0], origin[1], "red", 1, 1)


def draw_1st_layer_circles(radius=50, size=4):
    radius = 50
    list_of_degs = list_of_1st_layer_degs
    s = ''
    list_of_colors = list_of_1st_layer_colors
    for idx, (d, clr) in enumerate(zip(list_of_degs, list_of_colors)):
        color = clr_dict[clr]
        if idx in indexes[0]:
            a = 1
        else:
            a = alpha
        c = circle_f(origin, radius, d)
        s += circle.format(c[0], c[1], size, color, a, a)
    return s


def draw_1st_lines(radius_1=10, radius_2=40):
    list_of_degs = list_of_1st_layer_degs
    s = ''
    list_of_colors = list_of_1st_layer_colors
    for idx, (d, clr) in enumerate(zip(list_of_degs, list_of_colors)):
        # color = clr_dict[clr]
        if idx in indexes[0]:
            a = 1
        else:
            a = alpha
        location_1 = circle_f(origin, radius_1, d)
        location_2 = circle_f(origin, radius_2, d)
        s += line.format(location_1[0],
                         location_1[1],
                         location_2[0],
                         location_2[1],
                         sw,
                         a)
    return s


def draw_2nd_layer_circles(radius=110, size=3):
    list_of_degs = list_of_2nd_layer_degs
    s = ''
    list_of_colors = list_of_2nd_layer_colors
    for idx, (d, clr) in enumerate(zip(list_of_degs, list_of_colors)):
        color = clr_dict[clr]
        if idx in indexes[1]:
            a = 1
        else:
            a = alpha
        c = circle_f(origin, radius, d)
        s += circle.format(c[0], c[1], size, color, a, a)
    return s


def draw_2nd_lines(radius_1=60, radius_2=100):
    list_of_prev_degs = np.repeat(np.array(list_of_1st_layer_degs), 4)
    list_of_next_degs = list_of_2nd_layer_degs
    s = ''
    for idx, (d1, d2) in enumerate(zip(list_of_prev_degs, list_of_next_degs)):
        if idx in indexes[1]:
            a = 1
        else:
            a = alpha
        location_1 = circle_f(origin, radius_1, d1)
        location_2 = circle_f(origin, radius_2, d2)
        s += line.format(location_1[0],
                         location_1[1],
                         location_2[0],
                         location_2[1],
                         sw,
                         a)
    return s


def draw_3rd_layer_circles(radius=248, size=2.5):
    list_of_degs = list_of_3rd_layer_degs
    s = ""
    list_of_colors = list_of_3rd_layer_colors
    for idx, (d, clr) in enumerate(zip(list_of_degs, list_of_colors)):
        color = clr_dict[clr]
        if idx in indexes[2]:
            a = 1
        else:
            a = alpha
        c = circle_f(origin, radius, d)
        s += circle.format(c[0], c[1], size, color, a, a)
    return s


def draw_3rd_lines(radius_1=120, radius_2=240):
    list_of_prev_degs = np.repeat(np.array(list_of_2nd_layer_degs), 4)
    list_of_next_degs = list_of_3rd_layer_degs
    s = ''
    for idx, (d1, d2) in enumerate(zip(list_of_prev_degs, list_of_next_degs)):
        if idx in indexes[2]:
            a = 1
        else:
            a = alpha
        location_1 = circle_f(origin, radius_1, d1)
        location_2 = circle_f(origin, radius_2, d2)
        s += line.format(location_1[0],
                         location_1[1],
                         location_2[0],
                         location_2[1],
                         sw,
                         a)
    return s


def draw_separator():
    radius = dim[0] * 0.5
    a = circle_f(origin, radius, 90)
    b = circle_f(origin, radius, 90 + 120)
    c = circle_f(origin, radius, 90 + 240)
    s = ""
    for i in [a, b, c]:
        s += line.format(origin[0],
                         origin[1],
                         i[0],
                         i[1],
                         1,
                         1)
    return s


def draw_all():
    canvas = ""
    # canvas += draw_origin_cirle()
    canvas += draw_1st_layer_circles()
    canvas += draw_1st_lines()
    canvas += draw_2nd_layer_circles()
    canvas += draw_2nd_lines()
    canvas += draw_3rd_layer_circles()
    canvas += draw_3rd_lines()
    canvas += draw_separator()
    s = svg_str.format(dim[0], dim[1], canvas)
    return s


print(draw_all())
