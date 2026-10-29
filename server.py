"""Локальный MCP-сервер: кадр с карты захвата -> текст.

Основной OCR — RapidOCR (PP-OCRv5, модель eslav: русский/английский) на GPU
через DirectML; запасной — встроенный OCR Windows.

Настройки через переменные окружения:
  CAPTURE_DEVICE  индекс устройства DirectShow (по умолчанию 0)
  CAPTURE_WIDTH   ширина кадра (1920)
  CAPTURE_HEIGHT  высота кадра (1080)
  OCR_ENGINE      rapid | windows (rapid)
  OCR_LANG        язык для Windows OCR: ru, en-US ... (ru)
  OCR_GPU         1 = RapidOCR на видеокарте через DirectML (по умолчанию 1)
  KEEP_OPEN       1 = держать карту открытой между вызовами (быстрее,
                  но OBS не сможет её использовать). По умолчанию 0.
"""
import asyncio
import os
import threading

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
from mcp.server.mcpserver import Image, MCPServer

DEVICE = int(os.environ.get("CAPTURE_DEVICE", "0"))
WIDTH = int(os.environ.get("CAPTURE_WIDTH", "1920"))
HEIGHT = int(os.environ.get("CAPTURE_HEIGHT", "1080"))
OCR_ENGINE = os.environ.get("OCR_ENGINE", "rapid")
OCR_LANG = os.environ.get("OCR_LANG", "ru")
OCR_GPU = os.environ.get("OCR_GPU", "1") == "1"
KEEP_OPEN = os.environ.get("KEEP_OPEN", "0") == "1"

mcp = MCPServer("capture-ocr")
_cap_lock = threading.Lock()
_ocr_lock = threading.Lock()
_cap = None
_rapid = None


# ---------- захват ----------

def _open():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Не удалось открыть устройство захвата #{DEVICE}. "
            "Возможно, его занимает OBS или другая программа."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


def _grab():
    global _cap
    with _cap_lock:
        cap = _cap or _open()
        try:
            # первые кадры после открытия бывают пустыми или старыми
            for _ in range(1 if _cap else 5):
                cap.grab()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Устройство открылось, но кадр не получен (нет сигнала?).")
            return frame
        finally:
            if KEEP_OPEN:
                _cap = cap
            else:
                cap.release()


def _parse_region(region):
    if not region:
        return None
    x, y, w, h = (int(v) for v in region.split(","))
    return x, y, w, h


# ---------- OCR ----------
# Оба движка возвращают список (x, y, w, h, text, score) в координатах img.

def _rapid_engine():
    global _rapid
    if _rapid is None:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        _rapid = RapidOCR(params={
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec.ESLAV,
            # классификатор поворота на экранном тексте только вредит
            "Global.use_cls": False,
            "Global.max_side_len": 4000,
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.use_dml": OCR_GPU,
        })
    return _rapid


def _ocr_rapid(img):
    with _ocr_lock:
        r = _rapid_engine()(img)
    if r.boxes is None:
        return []
    items = []
    for box, text, score in zip(r.boxes, r.txts, r.scores):
        xs, ys = box[:, 0], box[:, 1]
        items.append((xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min(), text, float(score)))
    return items


async def _ocr_windows(img):
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.media.ocr import OcrEngine

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w = bgra.shape[:2]
    bmp = SoftwareBitmap(BitmapPixelFormat.BGRA8, w, h, BitmapAlphaMode.PREMULTIPLIED)
    bmp.copy_from_buffer(bgra.tobytes())
    engine = OcrEngine.try_create_from_language(Language(OCR_LANG))
    if engine is None:
        raise RuntimeError(f"OCR для языка '{OCR_LANG}' не установлен в Windows.")
    result = await engine.recognize_async(bmp)
    items = []
    for line in result.lines:
        rects = [w.bounding_rect for w in line.words]
        x0 = min(r.x for r in rects); y0 = min(r.y for r in rects)
        x1 = max(r.x + r.width for r in rects); y1 = max(r.y + r.height for r in rects)
        items.append((x0, y0, x1 - x0, y1 - y0, line.text, 1.0))
    return items


def _is_noise(text, score, min_score):
    t = text.strip()
    if not t or score < min_score:
        return True
    # одиночные символы — обычно иконки, распознанные как буквы
    return len(t) == 1 and not t.isdigit()


def _layout(items, with_boxes):
    """Собирает фрагменты в строки экрана: сверху вниз, внутри строки слева направо."""
    items = sorted(items, key=lambda it: it[1] + it[3] / 2)
    rows = []
    for it in items:
        cy = it[1] + it[3] / 2
        if rows:
            row = rows[-1]
            ry = sum(i[1] + i[3] / 2 for i in row) / len(row)
            rh = sum(i[3] for i in row) / len(row)
            if abs(cy - ry) < 0.5 * max(rh, it[3]):
                row.append(it)
                continue
        rows.append([it])
    lines = []
    for row in rows:
        row.sort(key=lambda it: it[0])
        if with_boxes:
            parts = [f"[{round(x)},{round(y)},{round(w)},{round(h)}] {t}" for x, y, w, h, t, _ in row]
        else:
            parts = [t for *_, t, _ in row]
        lines.append("   ".join(parts))
    return "\n".join(lines)


# ---------- инструменты ----------

@mcp.tool()
async def read_screen(
    region: str = "",
    scale: float = 1.0,
    with_boxes: bool = False,
    engine: str = "",
    min_score: float = 0.5,
) -> str:
    """Распознаёт текст на экране второго ПК (кадр с карты захвата).

    Строки выводятся в порядке расположения на экране; фрагменты, стоящие
    на одной высоте (разные колонки/окна), разделены тремя пробелами.

    region: необязательная область "x,y,w,h" в пикселях исходного кадра
            (1920x1080). Пусто = весь экран.
    scale: увеличение перед OCR. Для всего экрана оптимально 1.0; для
           небольшой области с мелким текстом можно 2-3.
    with_boxes: добавить к каждому фрагменту координаты [x,y,w,h] в пикселях кадра.
    engine: "rapid" (нейросеть, точнее на мелком тексте) или "windows".
            Пусто = по умолчанию (rapid).
    min_score: отбросить фрагменты с уверенностью ниже порога (0..1, только rapid).
    """
    frame = await asyncio.to_thread(_grab)
    reg = _parse_region(region)
    ox, oy = (reg[0], reg[1]) if reg else (0, 0)
    img = frame[oy:oy + reg[3], ox:ox + reg[2]] if reg else frame
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    eng = engine or OCR_ENGINE
    if eng == "windows":
        items = await _ocr_windows(img)
    elif eng == "rapid":
        items = await asyncio.to_thread(_ocr_rapid, img)
    else:
        raise ValueError(f"Неизвестный движок '{eng}', ожидается rapid или windows.")

    items = [
        (ox + x / scale, oy + y / scale, w / scale, h / scale, t, s)
        for x, y, w, h, t, s in items
        if not _is_noise(t, s, min_score)
    ]
    return _layout(items, with_boxes) if items else "(текст не найден)"


@mcp.tool()
async def screenshot(region: str = "", max_width: int = 1600) -> Image:
    """Возвращает снимок экрана второго ПК (кадр с карты захвата) как изображение.

    region: необязательная область "x,y,w,h" в пикселях исходного кадра.
    max_width: уменьшить до этой ширины (экономит токены).
    """
    frame = await asyncio.to_thread(_grab)
    reg = _parse_region(region)
    img = frame[reg[1]:reg[1] + reg[3], reg[0]:reg[0] + reg[2]] if reg else frame
    if img.shape[1] > max_width:
        k = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Image(data=buf.tobytes(), format="jpeg")


if __name__ == "__main__":
    if OCR_ENGINE == "rapid":
        _rapid_engine()  # загрузить модели заранее, чтобы первый вызов был быстрым
    mcp.run()
