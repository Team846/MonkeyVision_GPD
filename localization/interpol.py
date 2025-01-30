class InterpolationTable:
    def __init__(self, input_data, real_data):
        self.input_data = input_data
        self.real_data = real_data

    def interpolate(self, x):
        x_values = self.input_data
        y_values = self.real_data
        if x <= x_values[0]:
            x0, x1 = x_values[0], x_values[1]
            y0, y1 = y_values[0], y_values[1]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        elif x >= x_values[-1]:
            x0, x1 = x_values[-2], x_values[-1]
            y0, y1 = y_values[-2], y_values[-1]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        else:
            for i in range(len(x_values) - 1):
                if x_values[i] <= x <= x_values[i + 1]:
                    x0, x1 = x_values[i], x_values[i + 1]
                    y0, y1 = y_values[i], y_values[i + 1]
                    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class Interpol:
    INPUT_DISTANCES = [2.644, 2.873, 3.550, 4.275, 4.766, 5.333, 6.570]
    REAL_DISTANCES = [66.5, 73.0, 92.0, 113.5, 129.0, 152.5, 194.5]

    PureDistTable = InterpolationTable(INPUT_DISTANCES, REAL_DISTANCES)

    INPUT_ANG = [0.0, 3.2, 6.92, 10.4, 14.2, 21.25, 24.9, 33.0]
    D_FAC_ANG = [1.0, 1.004, 1.0196, 1.0409, 1.0864, 1.1367, 1.2061, 1.2700]

    AngularDistTable = InterpolationTable(INPUT_ANG, D_FAC_ANG)
