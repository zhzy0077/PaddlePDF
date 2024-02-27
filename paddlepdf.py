#!/usr/bin/python
import fitz  # PyMuPDF
import cv2
import numpy as np
import tqdm
from paddleocr import PaddleOCR
import logging
import argparse
import filetype
import os
from pathlib import Path
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import time

logging.disable(logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

def process_pdf(pdf_doc, output_pdf_path, use_gpu):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu)
    for page_number in tqdm.tqdm(range(pdf_doc.page_count)):
        page = pdf_doc.load_page(page_number)
        page.add_redact_annot(page.rect)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        image_list = page.get_images()
        if not image_list:
            continue
        pix = fitz.Pixmap(pdf_doc, image_list[0][0])
        cim = cv2.cvtColor(
            np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                (pix.height, pix.width, pix.n)
            ),
            cv2.COLOR_RGB2BGR,
        )
        text = ocr.ocr(cim)

        def conv(C):
            C[0] /= cim.shape[1]
            C[1] /= cim.shape[0]
            C[0] *= page.rect.x1
            C[1] *= page.rect.y1
            return C

        fn = "china-s"
        if not text[0]:
            continue
        for o in text[0]:
            R = fitz.Rect(conv(o[0][0]), conv(o[0][2]))
            word = o[1][0]
            fs = R.width / fitz.get_text_length(word, fn, 1)
            page.insert_text(
                (R.x0, R.y1),
                word,
                fontname=fn,
                fontsize=fs,
                render_mode=3,
            )
    pdf_doc.save(output_pdf_path, garbage=4, deflate=True)
    pdf_doc.close()
    logging.info(f'Saved file {output_pdf_path!r}.')

def process(input, output, use_gpu):
    kind = filetype.guess(input)
    if kind.mime.startswith("image"):
        img_doc = fitz.open(input)
        pdf_doc = fitz.open("pdf", img_doc.convert_to_pdf())
        process_pdf(pdf_doc, output, use_gpu)

    elif kind.mime == "application/pdf":
        pdf_doc = fitz.open(input)
        process_pdf(pdf_doc, output, use_gpu)

    else:
        raise ValueError("Unsupported file type")

class Handler(FileSystemEventHandler):
    def __init__(self, output, use_gpu) -> None:
        super().__init__()
        self._output = output
        self._gpu = use_gpu

    def on_created(self, event):
        logging.info(f'Detected file {event.src_path!r}.')
        file = Path(event.src_path).stem;
        process(event.src_path, self._output + "/" + file + ".pdf", self._gpu)

    def on_moved(self, event):
        logging.info(f'Detected file {event.dest_path!r}.')
        file = Path(event.dest_path).stem;
        process(event.src_path, self._output + "/" + file + ".pdf", self._gpu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program than adds hidden(but copiable) text layer to image pdf.",
        epilog="Copyright (C) 2024 Cao Yang. This is free software; distributed under GPLv3. There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.",
    )
    parser.add_argument("input_file", help="Input PDF/Image file")
    parser.add_argument("output_file", help="Output PDF file")
    parser.add_argument("--use_gpu", help="Use GPU", default=True, type=bool)
    parser.add_argument("--watch", action='store_true', help="Watch input_file as a folder", default=False)
    args = parser.parse_args()

    if args.watch:
        logging.info(f'start watching directory {args.input_file!r}')
        event_handler = Handler(args.output_file, args.use_gpu)
        observer = PollingObserver(timeout=10)
        observer.schedule(event_handler, args.input_file, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)
        finally:
            observer.stop()
            observer.join()

    process(args.input_file, args.output_file, args.use_gpu)
