"""
合同智能提取工具 - Gradio Web 界面
技术链路: PDF/图片 → Ollama视觉模型(qwen2.5-vl) → 结构化字段 → Excel台账

使用方式:
  开发: pip install -r requirements.txt && python app.py
  打包: pyinstaller --onedir --name 合同提取工具 app.py
"""

import gradio as gr
import requests
import json
import base64
import re
import time
from io import BytesIO
from datetime import datetime
from pathlib import Path

from PIL import Image
import openpyxl

# ===================== 配置区 =====================
OLLAMA_BASE = "http://localhost:11434"
MODEL_NAME = "qwen2.5-vl-7b"
EXCEL_PATH = "合同台账模板.xlsx"
MAX_IMAGE_LONGEST = 2048       # 图片最长边（像素），太大影响响应速度
MAX_PDF_PAGES = 3              # PDF 最多取前 N 页
MAX_RETRIES = 3
RETRY_DELAY_SEC = 3

HEADERS = [
    "合同标题", "甲方全称", "乙方全称", "合同总金额", "金额大写",
    "生效日期", "到期日期", "服务期", "付款条件", "违约责任",
    "签约日期", "合同标的", "处理时间", "状态",
]

SYSTEM_PROMPT = """你是合同信息提取专家。请从合同中提取以下13个字段，严格按照JSON格式返回（只返回JSON对象，不要包含其他任何文字）：

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
2. 金额带数字和单位（如"500,000元"或"伍拾万元整"）
3. 日期统一 YYYY-MM-DD 格式
4. 付款条件有多条用分号连接
5. 只输出JSON，不要解释"""


# ===================== 工具函数 =====================

def check_ollama():
    """检查 Ollama 服务和模型是否就绪"""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code != 200:
            return False, f"Ollama 返回异常: HTTP {r.status_code}"
        models = r.json().get("models", [])
        names = [m["name"] for m in models]
        # 模糊匹配（支持 qwen2.5-vl:7b / qwen2.5-vl-7b:latest 等变体）
        ok = any(MODEL_NAME.split(":")[0] in n for n in names)
        if not ok:
            return False, f"未找到模型 {MODEL_NAME}，请运行: ollama pull {MODEL_NAME}"
        return True, f"✅ Ollama 运行中 | 模型 {MODEL_NAME} 已就绪"
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


def merge_images_vertically(images):
    """多图垂直拼接为一张"""
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    w = max(img.width for img in images)
    h = sum(img.height for img in images)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
    return canvas


def image_to_data_url(img):
    """PIL.Image → data:image/png;base64,..."""
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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


def call_ollama_vision(data_url: str):
    """调用 Ollama 视觉模型，返回解析后的 dict"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 1024},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
            if resp.status_code != 200:
                time.sleep(RETRY_DELAY_SEC)
                continue
            body = resp.json()
            content = body.get("message", {}).get("content", "")
            result = safe_parse_json(content)
            if result:
                return result
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
    return None


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
    """处理单个文件 → (字段数据 dict, 日志文本)"""
    path = Path(file_path)
    ext = path.suffix.lower()
    name = path.name

    try:
        # ---- 1. 转图片 ----
        if ext == ".pdf":
            images = pdf_to_images(file_path)
            img = merge_images_vertically(images)
            if img is None:
                return None, f"{name}: PDF 无内容"
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
            img = resize_image(Image.open(file_path))
        else:
            return None, f"{name}: 不支持的格式"

        # ---- 2. 调 AI ----
        data_url = image_to_data_url(img)
        result = call_ollama_vision(data_url)

        if result is None:
            return None, f"{name}: AI 提取失败（已重试{MAX_RETRIES}次）"

        return result, f"{name}: ✅ 提取成功"

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
                    label="📋 提取结果（双击单元格可手动修正）",
                    interactive=True,
                    wrap=True,
                )

        # ---- 底部：日志 ----
        log_box = gr.Textbox(label="📝 处理日志", lines=7, max_lines=15)

        # ---- 隐藏状态（存文件名，写入时用） ----
        filenames_state = gr.State([])

        # ---- 事件绑定 ----
        demo.load(fn=check_ollama, outputs=[status])

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
    _ensure_excel(EXCEL_PATH)

    # 预检 Ollama（只打印，不阻塞启动）
    ok, msg = check_ollama()
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  合同智能提取工具")
    print(f"  状态: {msg}")
    print(f"  打开: http://127.0.0.1:7860")
    print(f"  退出: Ctrl+C")
    print(f"{sep}\n")

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        quiet=False,
    )
