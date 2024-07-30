"""Image processing module"""
import os
import cv2

class ImageProcess():
    """Image processing class"""
    def __init__(self, waitkey=1000):
        self.waitkey = waitkey

        # Black/white color code
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)

        # RGB color code
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)

        # Printer color code
        self.cyan = (0, 255, 255)
        self.magenta = (255, 0, 255)
        self.yellow = (255, 255, 0)
        
        self.face_color_code = [self.blue, self.cyan, self.magenta, self.green, self.red]

        # Font
        self.font = cv2.FONT_HERSHEY_DUPLEX

    def read_image(self, image_path):
        """Read image from path"""
        img = None
        if os.path.exists(image_path):
            extention = image_path.split(".")[-1]
            if extention in ["jpg", "png", "bmp"]:
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:
                print("Not supported image type!!!")
        else:
            print("No image!!!")
        return img

    def show_image(self, image):
        """Show image"""
        cv2.imshow("Result", image)
        k = cv2.waitkey(self.waitkey)
        return  k

    def save_image(self, image, path):
        """Save image"""
        cv2.imwrite(path, image)

    def close_image(self):
        """Close image"""
        cv2.destroyAllWindow()

    def insert_text(self, image, b):
        """Draw rectangle on image"""
        cx = b[0]
        cy = b[1] + 12
        text = "{:.4f}".format(b[4])
        return cv2.putText(image, text, (cx, cy), self.font, 0.5, self.white)

    def draw_rectangle(self, image, b):
        """Draw rectangle on image"""
        return cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

    def draw_circle(self, image, b1, b2, color):
        """Draw rectangle on image"""
        return cv2.circle(image, (b1, b2, 1, color, 4))

    def process_image(self, image, b):
        """Process image"""
        image = self.insert_text(image, b)
        image = self.draw_rectangle(image, b)
        for i in range(5):
            image = self.draw_circle(image, b[(2*i)+5], b[(2*i)+6], self.face_color_code[i])
        return image

    def visualize_image(self, image, path, save=False):
        """Show and or save image"""
        k = self.show_image(image)
        if k == ord('s') or save:
            self.save_image(image, path)
        self.close_image()

class ImageStream():
    """Image stream processing class"""
    def __init__(self, path, is_stream=False):
        self.is_stream = is_stream
        self.path = path
        self.image_process = ImageProcess()
        self._init_stream()
        self.image_generator = self._get_generator()

    def _init_stream(self):
        self.open_stream()

    def _read_stream(self):
        """Read and return image from stream"""
        ret, frame = self.stream.read()
        if not ret:
            frame = None
        yield frame

    def _read_image(self):
        """Read and return image"""
        for img_path in self.stream:
            yield self.image_process.read_image(img_path)

    def _get_generator(self):
        if self.is_stream:
            return self._read_stream
        else:
            return self._read_image

    def _open_videocap(self):
        """Open video stream"""
        self.stream = cv2.VideoCapture(self.path)

    def _close_videocap(self):
        """Close video stream"""
        self.stream.release()

    def _get_image_path(self):
        """Get image paths"""
        self.stream = []
        if isinstance(self.path, list):
            file_list = os.listdir()
            for file_path in file_list:
                if os.path.isfile(file_path):
                    self.stream.append(file_path)
        else:
            self.stream = [self.path]

    def open_stream(self):
        """Open image stream"""
        if self.is_stream:
            self._open_videocap()
        else:
            self._get_image_path()

    def close_stream(self):
        """Close image stream"""
        if self.is_stream:
            self.stream.release()

    def get_image(self):
        """Return image"""
        return next(self.image_generator)
