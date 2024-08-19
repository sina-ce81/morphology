import cv2
import numpy as np



image = cv2.imread('Line-circle.png', cv2.IMREAD_GRAYSCALE)

# باینری کردن تصویر
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)


# ایجاد کرنل‌ها برای شناسایی خطوط

kernel_h = np.ones((30,2)) # کرنل برای خطوط افقی

kernel_v = np.ones((2,30))  # کرنل برای خطوط عمودی

# کرنل برای خطوط مورب (45 درجه)
kernel_d1 = np.array([ 
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0], 
    [1, 0, 0, 0, 0]
], dtype=np.uint8) 

# کرنل برای خطوط مورب (-45 درجه) 
kernel_d2 =np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
], dtype=np.uint8)  

# شناسایی خطوط افقی
lines_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)

# شناسایی خطوط عمودی
lines_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

# شناسایی خطوط  45 درجه
lines_d1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_d1)

# شناسایی خطوط  -45 درجه
lines_d2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_d2)


# ترکیب نتایج
all_lines = cv2.bitwise_or(lines_h, lines_v) #ترکیب دو تصویر باینری
all_lines = cv2.bitwise_or(all_lines, lines_d1)
all_lines = cv2.bitwise_or(all_lines, lines_d2)



# ایجاد کرنل برای شناسایی دایره‌ها
kernel_circle = np.ones((7,7))

# شناسایی دایره‌ها
circles = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_circle)
cv2.imshow('cirlce',circles)
cv2.imwrite('circle.bmp',circles)

#پیدا کردن کانتور های دایره
contours_circles, _ = cv2.findContours(circles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
num_circle = len(contours_circles)
print('number of circles: ',num_circle) #اندازه کانتورهای پیدا شده برابر تعداد دایره می باشد


# حذف دایره‌ها از تصویر اصلی
lines_without_circles = cv2.subtract(all_lines, circles) #تفاضل دو تصویر باینری
cv2.imwrite('line.bmp',lines_without_circles)

#افزودن عملگر گسترش برای پر کردن فاصله ها تاثیر بهتری ندارد
# kernel = np.ones((3,3))
# lines_without_circles = cv2.morphologyEx(lines_without_circles,cv2.MORPH_DILATE,kernel)

#پیدا کردن کانتور های خطوط
contours_lines,_ = cv2.findContours(lines_without_circles,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
num_lines = len(contours_lines)
print('number of lines: ',num_lines) #اندازه کانتورهای پیدا شده برابر تعداد خطوط می باشد


cv2.imshow('lines',lines_without_circles)
cv2.waitKey(0)
