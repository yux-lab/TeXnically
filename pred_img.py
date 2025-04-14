from PIL import Image
from cli import LatexOCR

img = Image.open('dataset/data/test_img/09.png')
model = LatexOCR()
print(model(img))