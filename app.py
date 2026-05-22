"""
合同智能提取工具 - Gradio Web 界面
技术链路: PDF/图片 → Ollama视觉模型(qwen2.5-vl) → 结构化字段 → Excel台账

使用方式:
  开发: pip install -r requirements.txt && python app.py
  打包: pyinstaller --onedir --name 合同提取工具 app.py
"""

import sys
import os
import logging
import traceback
from datetime import datetime

# ===================== 日志系统（写入文件，防一闪而过） =====================
LOG_FILE = "contract_extractor.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# 捕获所有未处理的异常，写入日志文件
def global_excepthook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("程序崩溃", exc_info=(exc_type, exc_value, exc_traceback))
    # 写入专门错误文件
    with open("崩溃日志.txt", "w", encoding="utf-8") as f:
        f.write(f"时间: {datetime.now()}\n")
        f.write(f"类型: {exc_type.__name__}\n")
        f.write(f"信息: {exc_value}\n\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
        f.write("\n请将此文件发给开发者\n")

sys.excepthook = global_excepthook

# ===================== 清除系统代理（必须在 Gradio/httpx import 之前） =====================
# Windows/某些环境存在系统代理，会导致 httpx 无法访问 localhost
for _key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(_key, None)
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

import gradio as gr
import requests
import json
import base64
import re
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
import openpyxl

# ===================== Gradio schema 兼容补丁 (Python 3.9) =====================
# Gradio 4.x + Python 3.9 在 get_api_info 中有 TypeError，
# 当 schema 的 additionalProperties 为 False 时触发。
import gradio_client.utils as _gc_utils
import functools as _ft
if not hasattr(_gc_utils, '_patched'):
    _orig_fn = _gc_utils._json_schema_to_python_type
    @_ft.wraps(_orig_fn)
    def _patched_fn(schema, defs):
        if isinstance(schema, bool):
            return 'str'
        return _orig_fn(schema, defs)
    _gc_utils._json_schema_to_python_type = _patched_fn
    _gc_utils._patched = True

# ===================== 配置区 =====================
OLLAMA_BASE = "http://localhost:11434"
MODEL_NAME = "qwen2.5vl:3b"
EXCEL_PATH = "合同台账模板.xlsx"
MAX_IMAGE_LONGEST = 900        # 图片最长边（像素），3B模型友好
MAX_PDF_PAGES = 3              # PDF 最多取前 N 页
MAX_RETRIES = 3
RETRY_DELAY_SEC = 3

# ===================== 模型配置 =====================
TEXT_MODEL_NAME = "qwen2.5:3b"       # 文本提取模型（轻量，OCR后纯文本输入）

HEADERS = [
    "合同标题", "甲方全称", "乙方全称", "合同总金额", "金额大写",
    "生效日期", "到期日期", "服务期", "付款条件", "违约责任",
    "签约日期", "合同标的", "处理时间", "状态",
]

OCR_PROMPT = """请从这张合同图片中提取所有可见的文字内容，按原文顺序逐行输出。
不要改写，不要总结，不要分析，不要输出JSON。只输出原始文字。"""

EXTRACT_PROMPT = """以下是合同内容，请从中提取结构化字段：

{text}

严格按照JSON格式返回（只返回JSON对象，不要其他文字）：

{
  "合同标题": "",
  "甲方全称": "",
  "乙方全称": "",
  "合同总金额": "",
  "金额大写": "",
  "生效日期": "",
  "到期日期": "",
  "服务期": "",
  "付款条件": "",
  "违约责任": "",
  "签约日期": "",
  "合同标的": ""
}

要求：
1. 每项填入找到的具体内容，找不到填 null
2. 金额带数字和单位
3. 日期统一 YYYY-MM-DD 格式
4. 只输出JSON，不要解释"""

def check_ollama():
    """检查 Ollama 服务和模型是否就绪"""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code != 200:
            return False, f"Ollama 返回异常: HTTP {r.status_code}"
        models = r.json().get("models", [])
        names = [m["name"] for m in models]
        # 模糊匹配（去连字符、去冒号、统一小写比较）
        target = MODEL_NAME.replace("-", "").replace(":", "").replace("latest", "").lower()
        ok = any(target in n.replace("-", "").replace(":", "").replace("latest", "").lower() for n in names)
        if not ok:
            return False, f'未找到模型 {MODEL_NAME}\n已安装模型: {", ".join(names)}\n请运行: ollama pull {MODEL_NAME}'
        return True, f"✅ Ollama 运行中 | 模型 {MODEL_NAME} 可用"
    except requests.exceptions.ConnectionError:
        return False, "❌ Ollama 未启动，请先启动 Ollama 桌面端"


def resize_image(img):
    """等比例缩放，最长边不超过 MAX_IMAGE_LONGEST"""
    if max(img.size) > MAX_IMAGE_LONGEST:
        ratio = MAX_IMAGE_LONGEST / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    return img


def pdf_to_images(pdf_path: str):
    """PDF → PIL.Image 列表（前 MAX_PDF_PAGES 页，已缩放）"""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    pages = min(len(doc), MAX_PDF_PAGES)
    images = []
    for i in range(pages):
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))      # 2x 清晰度
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(resize_image(img))
    doc.close()
    return images


def merge_results(results: list) -> dict:
    """合并多页提取结果：第一页为主，后续页填补空字段"""
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    merged = results[0].copy()
    for r in results[1:]:
        for key in merged:
            val = merged.get(key)
            if not val or val == "null" or val is None or str(val).strip() == "":
                merged[key] = r.get(key, "")
    return merged


def image_to_base64(img):
    """PIL.Image → base64 string（Ollama 原生格式用裸 base64，不用 data URL）"""
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def safe_parse_json(text: str):
    """安全带兜底解析 JSON（多层 fallback）"""
    if not text or not text.strip():
        return None
    text = text.strip()

    # 1. 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 去掉 markdown 代码块
    cleaned = re.sub(r'^```(?:json)?\s*', '', text)
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. 正则提取第一个 {} 块
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _call_ollama(payload: dict) -> str:
    """通用 Ollama API 调用，带重试，返回响应文本"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
            if resp.status_code != 200:
                logging.warning(f"Ollama 返回 {resp.status_code}: {resp.text[:200]}")
                time.sleep(RETRY_DELAY_SEC)
                continue
            return resp.json().get("message", {}).get("content", "")
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
    return ""


def ocr_image(image_b64: str) -> str:
    """步骤1：视觉模型 OCR → 纯文本（只看字不提取）"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": OCR_PROMPT,
            "images": [image_b64],
        }],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 2048, "num_ctx": 2048},
    }
    return _call_ollama(payload)


def extract_fields(text: str) -> dict:
    """步骤2：文本模型从纯文字中提取结构化字段"""
    payload = {
        "model": TEXT_MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": EXTRACT_PROMPT.format(text=text),
        }],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    content = _call_ollama(payload)
    return safe_parse_json(content)


# ===================== Excel 操作 =====================

def _ensure_excel(path: str):
    """台账不存在时自动创建"""
    p = Path(path)
    if p.exists():
        return False, ""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    for i, h in enumerate(HEADERS, 1):
        c = ws.cell(row=1, column=i, value=h)
        c.font = openpyxl.styles.Font(bold=True)
        c.alignment = openpyxl.styles.Alignment(horizontal="center")
    wb.save(path)
    return True, f"已新建台账: {path}"


def append_row_to_excel(data: dict):
    """将一行数据写入 Excel 台账"""
    path = EXCEL_PATH
    _ensure_excel(path)
    try:
        wb = openpyxl.load_workbook(path)
        ws = wb.active
        row = ws.max_row + 1
        for i, key in enumerate(HEADERS[:-2], 1):
            val = data.get(key, "")
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False)
            ws.cell(row=row, column=i, value=str(val) if val is not None else "")
        ws.cell(row=row, column=len(HEADERS) - 1,
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ws.cell(row=row, column=len(HEADERS), value="成功")
        wb.save(path)
        return True, f"✅ 已写入台账，第 {row} 行"
    except PermissionError:
        return False, "❌ Excel 被其他程序占用，请关闭后重试"
    except Exception as e:
        return False, f"❌ 写入失败: {e}"


# ===================== 文件处理 =====================

def process_one_file(file_path: str):
    """处理单个文件 → (字段数据 dict, 日志文本)
    PDF 逐页调用 Ollama 后合并结果，避免大图爆显存
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    name = path.name

    try:
        # ---- 1. 转图片 ----
        if ext == ".pdf":
            images = pdf_to_images(file_path)
            if not images:
                return None, f"{name}: PDF 无内容"
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
            images = [resize_image(Image.open(file_path))]
        else:
            return None, f"{name}: 不支持的格式"

        # ---- 2. 步骤一：视觉模型 OCR 逐页看字 ----
        all_text = ""
        ocr_logs = []
        for i, img in enumerate(images):
            b64 = image_to_base64(img)
            text = ocr_image(b64)
            if text:
                all_text += f"\n=== 第{i+1}页 ===\n" + text
                ocr_logs.append(f"第{i+1}页✅")
            else:
                ocr_logs.append(f"第{i+1}页❌")

        if not all_text.strip():
            return None, f"{name}: OCR 失败 [{' '.join(ocr_logs)}]"

        # ---- 3. 步骤二：文本模型提取字段 ----
        result = extract_fields(all_text)
        if not result:
            return None, f"{name}: 字段提取失败（OCR成功但提取结果为空）"

        return result, f"{name}: OCR={' '.join(ocr_logs)} | 提取✅"

    except ImportError as e:
        return None, f"{name}: 缺少依赖 - {e}"
    except Exception as e:
        return None, f"{name}: ❌ 出错 - {e}"


def batch_process(files, progress=gr.Progress()):
    """批量处理上传的文件"""
    if not files:
        return [], "", ""

    # 先检查环境
    ok, status_msg = check_ollama()
    if not ok:
        return [], status_msg, []

    results = []
    logs = []
    filenames = []

    for i, f in enumerate(files):
        progress((i + 1) / len(files), desc=Path(f.name).name)
        data, log = process_one_file(f.name)
        if data:
            results.append(data)
            filenames.append(Path(f.name).name)
        logs.append(log)

    progress(1.0, desc="完成")

    # 构建表格：每行 = [合同标题, 甲方, ...]
    table = []
    for r in results:
        table.append([r.get(h, "") for h in HEADERS[:-2]])

    return table, "\n".join(logs), filenames


def write_table(table_data, filenames_state):
    """将当前表格中的每一行写入 Excel 台账"""
    if not table_data:
        return "没有数据可写入"
    out = []
    for i, row in enumerate(table_data):
        record = {}
        for j, key in enumerate(HEADERS[:-2]):
            val = row[j] if j < len(row) else ""
            record[key] = val
        ok, msg = append_row_to_excel(record)
        label = filenames_state[i] if i < len(filenames_state) else f"第{i+1}行"
        out.append(f"{label}: {msg}")
    return "\n".join(out)


# ===================== Gradio UI =====================

CSS = """
footer {display:none !important}
"""


def build_ui():
    with gr.Blocks(title="合同智能提取工具", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📄 合同智能提取工具
            上传合同 PDF 或图片 → AI 自动提取关键字段 → 一键写入 Excel 台账

            > **本地运行** · 使用 Ollama + qwen2.5-vl-7b · 数据不出本机
            """
        )

        # ---- 系统状态栏 ----
        status = gr.Textbox(label="系统状态", interactive=False)

        with gr.Row():
            # ---- 左侧：上传区 ----
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="📎 上传合同文件（PDF / 图片）",
                    file_count="multiple",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    type="filepath",
                    height=200,
                )
                with gr.Row():
                    extract_btn = gr.Button("⚡ 开始提取", variant="primary", size="lg")
                    save_btn = gr.Button("📥 写入台账", variant="secondary", size="lg")

            # ---- 右侧：结果表格 ----
            with gr.Column(scale=3):
                result_table = gr.Dataframe(
                    headers=HEADERS[:-2],
                    label="📋 提取结果",
                    interactive=False,
                    wrap=True,
                )

        # ---- 底部：日志 ----
        log_box = gr.Textbox(label="📝 处理日志", lines=7, max_lines=15)

        # ---- 隐藏状态（存文件名，写入时用） ----
        filenames_state = gr.State([])

        # ---- 事件绑定 ----
        # 只显示状态消息文本（check_ollama 返回 (bool, str)，取第二项）
        demo.load(fn=lambda: check_ollama()[1], outputs=[status])

        extract_btn.click(
            fn=batch_process,
            inputs=[file_input],
            outputs=[result_table, log_box, filenames_state],
        )

        save_btn.click(
            fn=write_table,
            inputs=[result_table, filenames_state],
            outputs=[log_box],
        )

        gr.Markdown(
            """
            ---
            **使用提示**
            1. 首次使用请先确认上方「系统状态」显示绿色 ✅
            2. 可一次上传多份合同，批量提取
            3. 提取结果支持手动编辑修正
            4. 修正后点击「写入台账」保存到 Excel
            """
        )

    return demo


# ===================== 入口 =====================

if __name__ == "__main__":
    try:
        _ensure_excel(EXCEL_PATH)

        ok, msg = check_ollama()
        logging.info(f"Ollama 状态: {msg}")

        sep = "=" * 50
        start_msg = (
            f"\n{sep}\n"
            f"  合同智能提取工具\n"
            f"  状态: {msg}\n"
            f"  打开: http://127.0.0.1:7860\n"
            f"  日志: {LOG_FILE}\n"
            f"  退出: Ctrl+C\n"
            f"{sep}\n"
        )
        print(start_msg)
        logging.info("启动 Gradio 服务...")

        # 绕过 Gradio 的 localhost 可达性检查（某些 Windows 环境下代理/TUN 驱动阻止回环）
        import gradio.networking as _gr_net
        _gr_net.url_ok = lambda url: True
        
        demo = build_ui()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            share=False,
            quiet=False,
            show_error=True,
        )
    except Exception as e:
        logging.exception("启动失败")
        with open("崩溃日志.txt", "w", encoding="utf-8") as f:
            f.write(f"时间: {datetime.now()}\n")
            f.write(f"错误: {e}\n\n")
            traceback.print_exc(file=f)
            f.write("\n请查看 contract_extractor.log 获取详细信息\n")
        print(f"\n❌ 程序启动失败，请查看「崩溃日志.txt」了解原因。")
        print(f"   或打开 {LOG_FILE} 查看完整日志。")
        # 暂停窗口，让用户看到错误
        try:
            input("\n按 Enter 退出...")
        except:
            pass
