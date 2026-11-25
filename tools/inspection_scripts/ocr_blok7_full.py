from rapidocr_onnxruntime import RapidOCR
import cv2

ocr = RapidOCR()
res, _ = ocr('output/debug_blok7_full.png')
if res:
    print(f"Total text blocks detected in Blok VII scan: {len(res)}")
    for line in res:
        bbox, text, score = line
        print(f"  {text:<45} score={float(score):.2f} y={int(bbox[0][1])}")
