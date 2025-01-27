class InterpolationTable:
    INPUT_DISTANCES = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    REAL_DISTANCES = [50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0]

    @staticmethod
    def interpolate(x):
        x_values = InterpolationTable.INPUT_DISTANCES
        y_values = InterpolationTable.REAL_DISTANCES
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
