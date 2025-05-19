import os
import json
import easyocr
import numpy as np
import cv2
import logging
import pytesseract
import warnings
from pdf2image import convert_from_path
from typing import List, Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image


class PDFTextExtractor:
    def __init__(self, languages: List[str] = ['en', 'ru'], use_tesseract: bool = False):
        """
        Инициализация OCR системы с поддержкой нескольких языков.

        :param languages: Список языков для распознавания
        :param use_tesseract: Флаг для использования Tesseract OCR вместо EasyOCR
        """
        self.use_tesseract = use_tesseract
        if not self.use_tesseract:
            self.reader = easyocr.Reader(languages)
        self._setup_logger()
        logging.info(f"PDFTextExtractor initialized with languages: {languages}, "
                     f"Tesseract: {'enabled' if use_tesseract else 'disabled'}")

    def _setup_logger(self):
        """Настройка логирования с ротацией логов."""
        logging.basicConfig(
            filename="pdf_text_extractor.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode='a'
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)

    def pdf_to_text(
            self,
            pdf_path: str,
            dpi: int = 300,
            output_json: Optional[str] = None,
            max_workers: int = 4,
            start_page: int = 1,
            end_page: Optional[int] = None
    ) -> Dict:
        """
        Извлекает текст из PDF с улучшенной обработкой изображений.

        :param pdf_path: Путь к PDF файлу
        :param dpi: Разрешение для конвертации PDF в изображения
        :param output_json: Путь для сохранения результатов в JSON
        :param max_workers: Максимальное количество потоков для обработки
        :param start_page: Начальная страница для обработки (1-based)
        :param end_page: Конечная страница для обработки (None = до конца)
        :return: Словарь с результатами обработки
        """
        logging.info(f"Начало обработки файла: {pdf_path}")
        print(f"\nНачало обработки файла: {os.path.basename(pdf_path)}")

        if not os.path.exists(pdf_path):
            error_msg = f"PDF файл не найден: {pdf_path}"
            logging.error(error_msg)
            return self._error_result(pdf_path, error_msg)

        result = {
            "source_file": os.path.basename(pdf_path),
            "file_size": f"{os.path.getsize(pdf_path) / 1024:.2f} KB",
            "timestamp": datetime.now().isoformat(),
            "status": "processing",
            "processing_time_sec": None,
            "pages": []
        }

        start_time = datetime.now()

        try:
            # Конвертируем PDF в изображения с оптимизацией памяти
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=start_page,
                last_page=end_page,
                thread_count=min(max_workers, 4),  # Ограничиваем потоки для экономии памяти
                fmt='jpeg',
                jpegopt={'quality': 80, 'progressive': True, 'optimize': True}
            )

            if not images:
                error_msg = "Не удалось конвертировать PDF в изображения"
                logging.error(error_msg)
                return self._error_result(pdf_path, error_msg)

            logging.info(f"Файл успешно прочитан, страниц для обработки: {len(images)}")
            print(f"Найдено страниц для обработки: {len(images)}")

            # Параллельная обработка страниц с контролем памяти
            with ThreadPoolExecutor(max_workers=min(max_workers, os.cpu_count() or 2)) as executor:
                futures = {
                    executor.submit(self._process_page, i, image): i
                    for i, image in enumerate(images, start=start_page)
                }

                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        page_result = future.result()
                        result["pages"].append(page_result)
                        logging.info(f"Страница {page_num} обработана успешно.")
                        print(f"Обработана страница {page_num}/{len(images)}")
                    except Exception as e:
                        error_msg = f"Ошибка обработки страницы {page_num}: {str(e)}"
                        logging.error(error_msg)
                        result["pages"].append({
                            "page_number": page_num,
                            "status": "error",
                            "error": error_msg
                        })

            result["status"] = "success"
            processing_time = (datetime.now() - start_time).total_seconds()
            result["processing_time_sec"] = processing_time
            logging.info(f"Обработка завершена за {processing_time:.2f} секунд.")
            print(f"\nОбработка завершена за {processing_time:.2f} секунд")

        except Exception as e:
            error_msg = f"Критическая ошибка обработки: {str(e)}"
            logging.error(error_msg)
            result.update(self._error_result(pdf_path, error_msg))
        finally:
            # Гарантированное сохранение результатов
            if output_json:
                try:
                    save_to_json(result, output_json)
                except Exception as e:
                    logging.error(f"Ошибка при сохранении результатов: {str(e)}")
                    print(f"Ошибка при сохранении: {str(e)}")
                    # Пробуем сохранить в текущую директорию
                    fallback_path = os.path.join(os.getcwd(), "ocr_result_fallback.json")
                    save_to_json(result, fallback_path)
                    print(f"Результаты сохранены в резервный файл: {fallback_path}")

        return result

    def _process_page(self, page_num: int, image: Image.Image) -> Dict:
        """Обрабатывает одну страницу изображения с освобождением памяти."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image_np = np.array(image)
            del image  # Освобождаем память

            image_np = self._preprocess_image(image_np)
            text = self._image_to_text(image_np)

            return {
                "page_number": page_num,
                "status": "success",
                "text": text,
                "text_length": len(text),
                "image_size": f"{image_np.shape[1]}x{image_np.shape[0]}"  # width x height
            }
        except Exception as e:
            logging.error(f"Ошибка обработки страницы {page_num}: {str(e)}")
            raise

    def _preprocess_image(self, image_np: np.ndarray) -> np.ndarray:
        """Предобрабатывает изображение для улучшения точности OCR."""
        try:
            # Конвертация в оттенки серого
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_np

            # Уменьшение размера для больших изображений
            h, w = gray.shape
            if h * w > 4000 * 4000:  # Если больше ~16 мегапикселей
                scale = 4000 / max(h, w)
                gray = cv2.resize(gray, None, fx=scale, fy=scale)

            # Удаление шума
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10,
                                                templateWindowSize=7,
                                                searchWindowSize=21)

            # Адаптивная бинаризация
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Улучшение резкости
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(binary, -1, kernel)

            return sharpened
        except Exception as e:
            logging.error(f"Ошибка предобработки изображения: {str(e)}")
            return image_np

    def _image_to_text(self, image_np: np.ndarray) -> str:
        """Извлекает текст с использованием выбранного OCR движка."""
        try:
            if self.use_tesseract:
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(image_np, config=custom_config)
            else:
                results = self.reader.readtext(image_np, detail=0, paragraph=True)
                text = "\n".join(results) if results else "No text detected"

            return text.strip()
        except Exception as e:
            logging.error(f"Ошибка OCR: {str(e)}")
            return f"OCR error: {str(e)}"

    def _error_result(self, pdf_path: str, error_msg: str) -> Dict:
        """Создает результат с ошибкой."""
        return {
            "source_file": os.path.basename(pdf_path),
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_msg,
            "pages": []
        }


def save_to_json(data: Dict, output_path: str):
    """Сохраняет данные в JSON файл с проверкой директории."""
    try:
        # Нормализация пути
        output_path = os.path.normpath(output_path)
        output_dir = os.path.dirname(output_path)

        # Создание директории, если нужно
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Сохранение с проверкой
        temp_path = output_path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Атомарная замена файла
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)

        # Проверка результата
        if not os.path.exists(output_path):
            raise RuntimeError(f"Файл не был создан: {output_path}")

        logging.info(f"Данные успешно сохранены в {output_path}")
        print(f"\nРезультаты сохранены в: {os.path.abspath(output_path)}")
    except Exception as e:
        logging.error(f"Ошибка сохранения JSON: {str(e)}")
        raise


def print_summary(result: Dict):
    """Выводит краткую статистику по результатам обработки."""
    print("\n" + "=" * 50)
    print(f"Результаты обработки файла: {result['source_file']}")
    print(f"Статус: {result['status']}")

    if result['status'] == 'success':
        print(f"\nОбработано страниц: {len(result['pages'])}")
        print(f"Общее время обработки: {result['processing_time_sec']:.2f} сек.")

        success_pages = [p for p in result['pages'] if p.get('status') == 'success']
        if success_pages:
            avg_text_len = sum(len(p['text']) for p in success_pages) / len(success_pages)
            print(f"Средняя длина текста на странице: {avg_text_len:.0f} символов")

        error_pages = [p for p in result['pages'] if p.get('status') == 'error']
        if error_pages:
            print(f"Страниц с ошибками: {len(error_pages)}")
    else:
        print(f"\nОшибка: {result['error']}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Извлечение текста из PDF с улучшенной обработкой OCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_pdf', help="Путь к PDF-документу для обработки")
    parser.add_argument('--output', help="Путь к выходному JSON-файлу", default='output.json')
    parser.add_argument('--dpi', type=int, help="DPI для конвертации", default=300)
    parser.add_argument('--lang', nargs='+', help="Языки для распознавания", default=['en', 'ru'])
    parser.add_argument('--workers', type=int, help="Количество потоков для обработки", default=4)
    parser.add_argument('--tesseract', action='store_true', help="Использовать Tesseract вместо EasyOCR")
    parser.add_argument('--start', type=int, help="Номер начальной страницы (1-based)", default=1)
    parser.add_argument('--end', type=int, help="Номер конечной страницы", default=None)

    args = parser.parse_args()

    # Проверка Tesseract
    if args.tesseract:
        try:
            pytesseract.get_tesseract_version()
            # Явное указание пути, если нужно
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        except EnvironmentError:
            print("Ошибка: Tesseract не установлен или не добавлен в PATH")
            print("Продолжаю с EasyOCR...")
            args.tesseract = False

    # Инициализация и обработка
    extractor = PDFTextExtractor(languages=args.lang, use_tesseract=args.tesseract)

    try:
        text_data = extractor.pdf_to_text(
            args.input_pdf,
            dpi=args.dpi,
            output_json=args.output,
            max_workers=args.workers,
            start_page=args.start,
            end_page=args.end
        )

        print_summary(text_data)

        # Дополнительная проверка сохранения
        if args.output and not os.path.exists(args.output):
            print("\nПредупреждение: основной файл результатов не найден!")
            fallback_path = os.path.join(os.getcwd(), "ocr_fallback_result.json")
            save_to_json(text_data, fallback_path)
            print(f"Резервная копия результатов сохранена в: {fallback_path}")

    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        logging.error(f"Критическая ошибка в основном потоке: {str(e)}")