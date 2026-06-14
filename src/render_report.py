"""
render_report.py — Markdown 보고서 → 스타일 HTML → PDF (headless Chrome).

report/report.md 를 학술 보고서 스타일 HTML 로 변환하고(한글 폰트, figure 캡션, A4),
figure 를 base64 로 임베드한 뒤 headless Chrome 으로 PDF 를 생성한다.
"""
from __future__ import annotations
import base64, os, re, subprocess
import markdown
from config import BASE_DIR, REPORT_DIR, FIG_DIR

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 19mm 17mm; }
* { box-sizing: border-box; }
body { font-family: 'Apple SD Gothic Neo','AppleGothic',sans-serif; color:#1a1a1a;
       line-height:1.62; font-size:10.5pt; }
h1 { font-size:21pt; color:#22335a; border-bottom:3px solid #f5a623; padding-bottom:8px;
     margin-top:6px; line-height:1.3; }
h2 { font-size:14pt; color:#22335a; margin-top:22px; border-left:5px solid #f5a623;
     padding-left:10px; }
h3 { font-size:11.5pt; color:#2a3b66; margin-top:16px; }
h2, h3 { page-break-after: avoid; }
p { margin:7px 0; text-align:justify; }
strong { color:#11203f; }
code { background:#f1f3f7; padding:1px 5px; border-radius:3px; font-size:9.5pt; }
table { border-collapse:collapse; width:100%; margin:12px 0; font-size:9.3pt; }
th { background:#22335a; color:#fff; padding:6px 8px; text-align:left; }
td { border:1px solid #d4d9e3; padding:5px 8px; }
tr:nth-child(even) td { background:#f7f9fc; }
img { max-width:100%; display:block; margin:10px auto; border:1px solid #e3e7ef; }
figure { margin:16px 0; page-break-inside:avoid; }
figcaption { text-align:center; font-size:9pt; color:#555; margin-top:4px; }
blockquote { border-left:4px solid #f5a623; background:#fff8ec; margin:10px 0; padding:8px 14px;
             color:#5a4a2a; }
.lead { font-size:11pt; color:#444; }
hr { border:none; border-top:1px solid #dde2ec; margin:18px 0; }
ul,ol { margin:6px 0 6px 4px; } li { margin:3px 0; }
"""


def _embed_images(html):
    def repl(m):
        src = m.group(1)
        fn = os.path.basename(src)
        path = os.path.join(FIG_DIR, fn)
        if os.path.exists(path):
            b64 = base64.b64encode(open(path, "rb").read()).decode()
            return f'src="data:image/png;base64,{b64}"'
        return m.group(0)
    return re.sub(r'src="([^"]+)"', repl, html)


def render(md_path=None, out_pdf=None):
    md_path = md_path or os.path.join(REPORT_DIR, "report.md")
    out_pdf = out_pdf or os.path.join(REPORT_DIR, "TeamG_Final_Report.pdf")
    md_text = open(md_path, encoding="utf-8").read()
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "attr_list", "md_in_html"])
    body = _embed_images(body)
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    html_path = os.path.join(REPORT_DIR, "report.html")
    open(html_path, "w", encoding="utf-8").write(html)

    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", "--no-margins",
                    f"file://{html_path}"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    return out_pdf, html_path


if __name__ == "__main__":
    pdf, html = render()
    sz = os.path.getsize(pdf) / 1024
    print(f"PDF 생성: {pdf} ({sz:.0f} KB)")
    print(f"HTML: {html}")
