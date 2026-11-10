from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="ch",                          # "ch" for Chinese, "japan", "korean", "fr", etc.
    use_doc_orientation_classify=False, # turn on for rotated document photos
    use_doc_unwarping=False,            # turn on for curved/warped pages
    use_textline_orientation=False,     # turn on if text lines may be upside down
)

results = ocr.predict("4.jpg")   # also accepts a PDF, a folder, a URL, or a numpy array

for res in results:
    res.print()                      # print everything
    res.save_to_img("output")        # image with boxes drawn
    res.save_to_json("output")       # structured results

    # Get the text yourself
    for text, score in zip(res["rec_texts"], res["rec_scores"]):
        print(f"{score:.2f}  {text}")