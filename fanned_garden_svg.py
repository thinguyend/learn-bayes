import numpy as np

d = 180
dim = (520, 250)
origin = (dim[0] / 2, dim[1]*0.9)
alpha = 0.3
sw = 2

svg_str = '<svg width="{}" height="{}">{}</svg>'
circle = '<circle cx="{}" cy="{}" ' +\
    'r="{}" stroke="black" stroke-width="1.5" fill="{}" ' +\
    'stroke-opacity="{}" fill-opacity="{}" />'
line = '<line x1="{}" y1="{}" x2="{}" y2="{}" ' +\
    'stroke="black" stroke-width="{}" stroke-opacity="{}" />'

clr_dict = {"B": "blue", "W": "white"}


def deg_to_rad(deg):
    return deg * np.pi / 180


def get_1st_layer_degs(deg=d):
    x, z = 50, 15
    list_of_degs = []
    deg -= z
    list_of_degs.append(deg)
    for i in range(3):
        deg -= x
        list_of_degs.append(deg)
    return list_of_degs


def get_2nd_layer_degs(deg=d):
    list_of_degs = []
    x, y, z = 9, 16, 12
    deg = deg - z
    for j in range(4):
        list_of_degs.append(deg)
        for i in range(3):
            deg -= x
            list_of_degs.append(deg)
        deg -= y
    return list_of_degs


def get_3rd_layer_degs(deg=d):
    list_of_degs = []
    x, y = 2.5, 4
    for i in range(16):
        list_of_degs.append(deg)
        for i in range(3):
            deg -= x
            list_of_degs.append(deg)
        deg -= y
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


def draw_1st_layer_circles(radius=50, size=4, all=True):
    radius = 50
    list_of_degs = list_of_1st_layer_degs
    s = ''
    list_of_colors = list_of_1st_layer_colors
    for idx, (d, clr) in enumerate(zip(list_of_degs, list_of_colors)):
        color = clr_dict[clr]
        a = 1
        if all is False:
            a = alpha
            if idx in indexes[0]:
                a = 1
        c = circle_f(origin, radius, d)
        s += circle.format(c[0], c[1], size, color, a, a)
    return s


def draw_1st_lines(radius_1=10, radius_2=40, sw=sw, all=True):
    list_of_degs = list_of_1st_layer_degs
    s = ''
    list_of_colors = list_of_1st_layer_colors
    for idx, (d, clr) in enumerate(zip(list_of_degs, list_of_colors)):
        # color = clr_dict[clr]
        a = 1
        if all is False:
            a = alpha
            if idx in indexes[0]:
                a = 1
        location_1 = circle_f(origin, radius_1, d)
        location_2 = circle_f(origin, radius_2, d)
        s += line.format(location_1[0],
                         location_1[1],
                         location_2[0],
                         location_2[1],
                         sw,
                         a)
    return s


def draw_2nd_layer_circles(radius=110, size=4, all=True):
    list_of_degs = list_of_2nd_layer_degs
    s = ''
    list_of_colors = list_of_2nd_layer_colors
    for idx, (d, clr) in enumerate(zip(list_of_degs, list_of_colors)):
        color = clr_dict[clr]
        a = 1
        if all is False:
            a = alpha
            if idx in indexes[1]:
                a = 1
        c = circle_f(origin, radius, d)
        s += circle.format(c[0], c[1], size, color, a, a)
    return s


def draw_2nd_lines(radius_1=60, radius_2=100, sw=sw, all=True):
    list_of_prev_degs = np.repeat(np.array(list_of_1st_layer_degs), 4)
    list_of_next_degs = list_of_2nd_layer_degs
    s = ''
    for idx, (d1, d2) in enumerate(zip(list_of_prev_degs, list_of_next_degs)):
        a = 1
        if all is False:
            a = alpha
            if idx in indexes[1]:
                a = 1
        location_1 = circle_f(origin, radius_1, d1)
        location_2 = circle_f(origin, radius_2, d2)
        s += line.format(location_1[0],
                         location_1[1],
                         location_2[0],
                         location_2[1],
                         sw,
                         a)
    return s


def draw_3rd_layer_circles(radius=204, size=4, all=True):
    list_of_degs = list_of_3rd_layer_degs
    s = ""
    list_of_colors = list_of_3rd_layer_colors
    for idx, (d, clr) in enumerate(zip(list_of_degs, list_of_colors)):
        color = clr_dict[clr]
        a = 1
        if all is False:
            a = alpha
            if idx in indexes[2]:
                a = 1
        c = circle_f(origin, radius, d)
        s += circle.format(c[0], c[1], size, color, a, a)
    return s


def draw_3rd_lines(radius_1=120, radius_2=200, sw=sw, all=True):
    list_of_prev_degs = np.repeat(np.array(list_of_2nd_layer_degs), 4)
    list_of_next_degs = list_of_3rd_layer_degs
    s = ''
    for idx, (d1, d2) in enumerate(zip(list_of_prev_degs, list_of_next_degs)):
        a = 1
        if all is False:
            a = alpha
            if idx in indexes[2]:
                a = 1
        location_1 = circle_f(origin, radius_1, d1)
        location_2 = circle_f(origin, radius_2, d2)
        s += line.format(location_1[0],
                         location_1[1],
                         location_2[0],
                         location_2[1],
                         sw,
                         a)
    return s


def draw_all():
    canvas = ""
    # canvas += draw_origin_cirle()
    canvas += draw_1st_layer_circles(all=False)
    canvas += draw_1st_lines(all=False)
    canvas += draw_2nd_layer_circles(all=False)
    canvas += draw_2nd_lines(all=False)
    canvas += draw_3rd_layer_circles(all=False)
    canvas += draw_3rd_lines(all=False)
    s = svg_str.format(dim[0], dim[1], canvas)
    return s


print(draw_all())
# print(list_of_1st_layer_degs)
# print(list_of_2nd_layer_degs)
