from PyPDF2 import PdfReader, PdfWriter

reader = PdfReader("target_alpha_eta.pdf")
writer = PdfWriter()

page = reader.pages[0]
# 原始裁剪框
box = page.mediabox
# 上方减少 50 单位（具体数值自己调）
box.upper_right = (box.upper_right[0], box.upper_right[1] - 90)
page.mediabox = box

writer.add_page(page)
with open("target_alpha_eta_cropped.pdf", "wb") as f:
    writer.write(f)
