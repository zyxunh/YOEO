import io
from PIL import Image

img = Image.open(fh, mode='r')
roi_img = img.crop(box)

img_byte_arr = io.BytesIO()
roi_img.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()