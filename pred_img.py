from PIL import Image
from cli import LatexOCR
import time

img = Image.open('dataset/data/test_img/0181880.png')
model = LatexOCR()

start_time = time.time()
result = model(img)
end_time = time.time()
elapsed_time = end_time - start_time

print("Result:", result)
print(f"Time: {elapsed_time:.2f} 秒")