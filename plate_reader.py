import cv2
import numpy as np

def preprocess_image(img):
    # Tăng cường độ tương phản bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Chuyển sang ảnh xám và làm mờ
    grayscale = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    
    return blurred

def detect_plate_canny(img):
    # Phát hiện cạnh với Canny (thử nhiều ngưỡng khác nhau)
    edged = cv2.Canny(img, 30, 200)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged

def detect_plate_adaptive(img):
    # Phát hiện biển số bằng adaptive threshold (hiệu quả với ảnh mờ)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def order_points(pts):
    # Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    
    # Điểm có tổng tọa độ nhỏ nhất là top-left, lớn nhất là bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Điểm có hiệu tọa độ nhỏ nhất là top-right, lớn nhất là bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    # Lấy các điểm và tính toán kích thước của ảnh kết quả
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Tính chiều rộng mới (max của khoảng cách giữa bottom-right và bottom-left hoặc top-right và top-left)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Tính chiều cao mới (max của khoảng cách giữa top-right và bottom-right hoặc top-left và bottom-left)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Tạo ma trận đích và tính ma trận biến đổi phối cảnh
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def find_plate_contours(edged, original_img):
    # Tìm contours và lọc theo diện tích và tỷ lệ khung hình
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        
        # Kiểm tra có phải hình chữ nhật không và có tỷ lệ khung hình hợp lý cho biển số
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Tỷ lệ khung hình điển hình của biển số xe (thay đổi tùy quốc gia)
            if 2.0 < aspect_ratio < 5.0:
                plate_contour = approx
                break
                
    return plate_contour

def main():
    # Đọc ảnh và thay đổi kích thước
    img = cv2.imread('cach-lap-bien-so-xe-o-to-chac-chan-khong-roi-r6o8ilr.jpg')
    if img is None:
        print("Không thể đọc ảnh")
        return
    
    img = cv2.resize(img, (1200, 900))
    
    # Tiền xử lý ảnh
    processed_img = preprocess_image(img)
    
    # Thử cả hai phương pháp phát hiện biển số
    edged_canny = detect_plate_canny(processed_img)
    edged_adaptive = detect_plate_adaptive(processed_img)
    
    # Tìm biển số với cả hai phương pháp
    plate_contour_canny = find_plate_contours(edged_canny, img)
    plate_contour_adaptive = find_plate_contours(edged_adaptive, img)
    
    # Ưu tiên contour từ phương pháp adaptive nếu tìm thấy
    plate_contour = plate_contour_adaptive if plate_contour_adaptive is not None else plate_contour_canny
    
    # Hiển thị kết quả
    if plate_contour is None:
        print("Không tìm thấy biển số xe.")
        cv2.imshow("Original Image", img)
        cv2.waitKey(0)
    else:
        # Vẽ contour lên ảnh gốc
        cv2.drawContours(img, [plate_contour], -1, (0, 255, 0), 3)
        
        # Reshape contour và áp dụng phép biến đổi phối cảnh
        pts = plate_contour.reshape(4, 2)
        warped = four_point_transform(img, pts)
        
        # Hiển thị kết quả
        cv2.imshow("Detected Plate", img)
        cv2.imshow("Corrected Number Plate", warped)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()