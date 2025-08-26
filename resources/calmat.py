import cv2
import numpy as np
from fpdf import FPDF

image = np.zeros((800, 1150, 3), dtype=np.uint8)

image[:] = (255, 255, 255)

num_ticks_x = 19
num_ticks_y = int(800 / 1150 * num_ticks_x)
tick_length = 15
tick_color = (0, 0, 0)
thickness = 4

center_x = 1150 // 2
center_y = 800 // 2

cv2.line(image, (0, center_y), (1149, center_y), (0, 0, 255), 1)
cv2.line(image, (center_x, 0), (center_x, 799), (0, 0, 255), 1)

x_positions = np.linspace(-center_x, 1149 - center_x, num_ticks_x, dtype=int)
for x in x_positions:
    px = center_x + x + 1
    if 10 <= px < 1150:
        cv2.line(
            image,
            (px, center_y + tick_length),
            (px, center_y - tick_length),
            tick_color,
            thickness,
        )

y_positions = np.linspace(-center_y, 799 - center_y, num_ticks_y, dtype=int)
for y in y_positions:
    py = center_y + y + 1
    if 10 <= py < 800:
        cv2.line(
            image,
            (center_x - tick_length, py),
            (center_x + tick_length, py),
            tick_color,
            thickness,
        )

image = cv2.resize(image, (842, 595))
cv2.imwrite("calmat.png", image)

pdf = FPDF(orientation="L", unit="pt", format="A4")
pdf.add_page()
img_width, img_height = 842, 595
pdf.image("calmat.png", 0, 0, img_width, img_height)
pdf.output("calmat.pdf")
