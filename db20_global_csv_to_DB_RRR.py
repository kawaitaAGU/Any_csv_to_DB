import streamlit as st
import pandas as pd
import io
import requests
from PIL import Image
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.utils import ImageReader
import time
from pathlib import Path
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import re

# ---- フォント設定（IPAex を優先、無ければCIDフォントへフォールバック）----
def _setup_font():
    here = Path(__file__).parent
    candidates = [
        here / "fonts" / "IPAexGothic.ttf",
        here / "IPAexGothic.ttf",
        Path.cwd() / "fonts" / "IPAexGothic.ttf",
        Path.cwd() / "IPAexGothic.ttf",
    ]
    for p in candidates:
        if p.exists():
            pdfmetrics.registerFont(TTFont("Japanese", str(p)))
            return "Japanese"
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    return "HeiseiKakuGo-W5"

JAPANESE_FONT = _setup_font()

st.set_page_config(page_title="🔍 学生指導用データベース", layout="wide")
st.title("🔍 学生指導用データベース")

# ===== 文字・改行ユーティリティ =====
def _strip(s):
    if s is None:
        return ""
    return str(s).replace("\ufeff", "").strip()

def _norm_space(s):
    return re.sub(r"[\u3000 \t\r\n]+", "", _strip(s))

def _normalize_newlines(text: str, newline: str = "\n") -> str:
    if text is None:
        return ""
    t = re.sub(r"\r\n|\r", "\n", str(text))
    if newline == "\r\n":
        t = t.replace("\n", "\r\n")
    return t

# ===== 列名正規化 & 取りこぼし救済 =====
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """BOM/空白/改行を除去し、よくある別名を正式名へ寄せる。
       さらに 'A18歯科理工学' のような 連結列 → 問題番号ID/科目分類 を自動抽出。"""
    df = df.copy()
    # 列名クレンジング
    orig_to_clean = {c: _norm_space(c) for c in df.columns}
    df.rename(columns=orig_to_clean, inplace=True)

    # よくある別名 → 正式名
    alias = {
        "問題文":  ["設問", "問題", "本文"],
        "選択肢1": ["選択肢Ａ","選択肢a","A","ａ"],
        "選択肢2": ["選択肢Ｂ","選択肢b","B","ｂ"],
        "選択肢3": ["選択肢Ｃ","選択肢c","C","ｃ"],
        "選択肢4": ["選択肢Ｄ","選択肢d","D","ｄ"],
        "選択肢5": ["選択肢Ｅ","選択肢e","E","ｅ"],
        "正解":    ["解答","答え","ans","answer","正答"],
        "科目分類": ["分類","科目","カテゴリ","カテゴリー"],
        "リンクURL": ["画像URL","画像リンク","リンク","画像Link","URL","url"],
    }
    colset = set(df.columns)
    for canon, cands in alias.items():
        if canon in colset:  # 既にある
            continue
        for c in cands:
            if c in colset:
                df.rename(columns={c: canon}, inplace=True)
                colset.add(canon)
                break

    # --- 取りこぼし救済：問題番号＋分類が連結された列を抽出 ---
    # 1) ヘッダーに '問題番号' を含む列
    candidate_cols = [c for c in df.columns if "問題番号" in c]
    # 2) 値が 'A12歯科理工学' のような「英字+数字+日本語」の列
    if not candidate_cols:
        for c in df.columns:
            series = df[c].astype(str)
            if series.str.contains(r"^[A-Za-z]\d+[\u4e00-\u9fff\u3040-\u30ff]+", regex=True, na=False).any():
                candidate_cols.append(c)
                break

    # 問題番号ID/科目分類の抽出（既存列がなければ作る）
    if candidate_cols:
        src = candidate_cols[0]
        # もしヘッダー自体に “A00歯科理工学” などが埋まっていれば、そこから既定分類を拾う
        m_head = re.search(r"([A-Za-z]\d+)?([\u4e00-\u9fff\u3040-\u30ff]+)", src)
        default_cat = m_head.group(2) if m_head else ""

        def split_code_cat(val: str):
            s = _strip(val)
            if not s:
                return "", default_cat
            m = re.match(r"([A-Za-z]\d+)([\u4e00-\u9fff\u3040-\u30ff]+)", s)
            if m:
                return m.group(1), m.group(2)
            # 数字＋日本語のケース（例：18歯科理工学）
            m2 = re.match(r"([0-9]+)([\u4e00-\u9fff\u3040-\u30ff]+)", s)
            if m2:
                return m2.group(1), m2.group(2)
            # 日本語のみ or それ以外
            return "", s

        if "問題番号ID" not in df.columns or "科目分類" not in df.columns:
            codes = []
            cats = []
            for v in df[src].astype(str):
                code, cat = split_code_cat(v)
                codes.append(code)
                cats.append(cat)
            if "問題番号ID" not in df.columns:
                df["問題番号ID"] = codes
            if "科目分類" not in df.columns:
                df["科目分類"] = cats

    return df

def safe_get(row: pd.Series | dict, keys, default=""):
    """Series/辞書から安全に値を取得（NaN, 空白, 別名を考慮）"""
    if isinstance(row, pd.Series):
        row = row.to_dict()
    for k in keys:
        if k in row:
            v = row.get(k)
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            s = str(v).strip() if v is not None else ""
            if s:
                return s
    return default

def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = [
        "問題文","選択肢1","選択肢2","選択肢3","選択肢4","選択肢5",
        "正解","科目分類","問題番号ID","リンクURL"
    ]
    out = df.copy()
    for c in need:
        if c not in out.columns:
            out[c] = ""
    return out

# ======================
# 入力CSV（Drag & Drop）
# ======================
uploaded = st.file_uploader("CSVをドラッグ＆ドロップ（または選択）", type=["csv"])
if uploaded is None:
    st.info("任意のCSVをアップロードしてください（ヘッダあり推奨）。")
    st.stop()

try:
    df = pd.read_csv(uploaded, dtype=str, encoding="utf-8-sig").fillna("")
except Exception:
    # 文字コード違いなどの保険
    df = pd.read_csv(uploaded, dtype=str, encoding="utf-8", errors="ignore").fillna("")
df = normalize_columns(df)

# ===== 検索 =====
query = st.text_input("問題文・選択肢・分類・問題番号で検索:")
st.caption("💡 検索語を `&` でつなげるとAND検索（例: レジン & 理工）")

if not query:
    st.stop()

keywords = [kw.strip() for kw in query.split("&") if kw.strip()]

def row_text(r: pd.Series) -> str:
    parts = [
        safe_get(r, ["問題文","設問","問題","本文"]),
        *[safe_get(r, [f"選択肢{i}"]) for i in range(1,6)],
        safe_get(r, ["正解","正答","解答","答え"]),
        safe_get(r, ["科目分類","分類","科目"]),
        safe_get(r, ["問題番号ID","問題番号"]),
        safe_get(r, ["リンクURL","URL","画像URL","画像リンク","リンク"]),
    ]
    return " ".join([p for p in parts if p])

df_filtered = df[df.apply(
    lambda row: all(kw.lower() in row_text(row).lower() for kw in keywords),
    axis=1
)]
df_filtered = df_filtered.reset_index(drop=True)

st.info(f"{len(df_filtered)}件ヒットしました")

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
file_prefix = f"{(query if query else '検索なし')}{timestamp}"

# ===== CSV ダウンロード =====
csv_buffer = io.StringIO()
ensure_output_columns(df_filtered).to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 ヒット結果をCSVダウンロード",
    data=csv_buffer.getvalue(),
    file_name=f"{file_prefix}.csv",
    mime="text/csv"
)

# ===== GoodNotes 用 CSV（Front/Back） =====
def _gn_clean(s: str) -> str:
    return _strip(s).replace("　", "")

def _gn_make_front_back(row: pd.Series,
                        numbering: str = "ABC",
                        add_labels: bool = True,
                        add_meta: bool = True) -> tuple[str, str]:
    q = _gn_clean(safe_get(row, ["問題文","設問","問題","本文"]))
    choices = [
        _gn_clean(safe_get(row, ["選択肢1"])),
        _gn_clean(safe_get(row, ["選択肢2"])),
        _gn_clean(safe_get(row, ["選択肢3"])),
        _gn_clean(safe_get(row, ["選択肢4"])),
        _gn_clean(safe_get(row, ["選択肢5"])),
    ]
    labels = ["A","B","C","D","E"] if numbering == "ABC" else ["1","2","3","4","5"]
    choice_lines = [f"{labels[i]}. {_normalize_newlines(txt)}" for i, txt in enumerate(choices) if txt]

    front = _normalize_newlines(q)
    if choice_lines:
        front = front + "\n\n" + "\n".join(choice_lines)

    ans = _gn_clean(safe_get(row, ["正解","正答","解答","答え"]))
    back = f"正解: {ans}" if add_labels else ans

    if add_meta:
        cat = _gn_clean(safe_get(row, ["科目分類","分類","科目"]))
        code = _gn_clean(safe_get(row, ["問題番号ID","問題番号"]))
        extra = "\n".join([s for s in (cat, code) if s])
        if extra:
            back = back + "\n\n" + _normalize_newlines(extra)

    back = _normalize_newlines(back)
    return front, back

def dataframe_to_goodnotes_bytes(df: pd.DataFrame) -> bytes:
    base = ensure_output_columns(df)
    fronts, backs = [], []
    for _, row in base.iterrows():
        f, b = _gn_make_front_back(row)
        fronts.append(f); backs.append(b)
    out = pd.DataFrame({"Front": fronts, "Back": backs})
    # セル内部改行はLF
    for c in out.columns:
        out[c] = out[c].map(lambda v: _normalize_newlines(v, "\n"))
    buf = io.StringIO()
    buf.write("\ufeff")  # BOM
    out.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")

st.download_button(
    label="📥 GoodNotes用CSV（Front/Back）をダウンロード",
    data=dataframe_to_goodnotes_bytes(df_filtered),
    file_name=f"{file_prefix}_goodnotes.csv",
    mime="text/csv",
)

# ===== TXT 整形 =====
def convert_google_drive_link(url):
    if "drive.google.com" in url and "/file/d/" in url:
        try:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=view&id={file_id}"
        except Exception:
            return url
    return url

def wrap_text(text: str, max_width: float, font_name: str, font_size: int):
    s = "" if text is None else str(text)
    if s == "":
        return [""]
    lines, buf = [], ""
    for ch in s:
        if stringWidth(buf + ch, font_name, font_size) <= max_width:
            buf += ch
        else:
            lines.append(buf)
            buf = ch
    if buf:
        lines.append(buf)
    return lines

def wrapped_lines(prefix: str, value: str, usable_width: float, font: str, size: int):
    return wrap_text(f"{prefix}{value}", usable_width, font, size)

def format_record_to_text(row: pd.Series) -> str:
    q = safe_get(row, ["問題文","設問","問題","本文"])
    parts = [f"問題文: {q}"]
    for i in range(1, 6):
        choice = safe_get(row, [f"選択肢{i}"])
        if choice:
            parts.append(f"選択肢{i}: {choice}")
    parts.append(f"正解: {safe_get(row, ['正解','正答','解答','答え'])}")
    parts.append(f"分類: {safe_get(row, ['科目分類','分類','科目'])}")
    code = safe_get(row, ["問題番号ID","問題番号"])
    if code:
        parts.append(f"問題番号: {code}")
    link = safe_get(row, ["リンクURL","画像URL","画像リンク","リンク","URL"])
    if link:
        parts.append(f"画像リンク: {convert_google_drive_link(link)}（PDFに画像表示）")
    return "\n".join(parts)

# ===== TXT ダウンロード =====
txt_buffer = io.StringIO()
for _, row in df_filtered.iterrows():
    txt_buffer.write(format_record_to_text(row))
    txt_buffer.write("\n\n" + "-"*40 + "\n\n")
st.download_button(
    label="📄 ヒット結果をTEXTダウンロード",
    data=txt_buffer.getvalue(),
    file_name=f"{file_prefix}.txt",
    mime="text/plain"
)

# ===== PDF 作成 =====
def create_pdf(records, progress=None, status=None, start_time=None):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    c.setFont(JAPANESE_FONT, 12)
    width, height = A4

    top_margin, bottom_margin = 40, 60
    left_margin, right_margin = 40, 40
    usable_width = width - left_margin - right_margin
    page_usable_h = (height - top_margin) - bottom_margin
    line_h = 18
    y = height - top_margin

    total = len(records)

    def new_page():
        nonlocal y
        c.showPage()
        c.setFont(JAPANESE_FONT, 12)
        y = height - top_margin

    def draw_wrapped_lines(lines):
        nonlocal y
        for ln in lines:
            c.drawString(left_margin, y, ln)
            y -= line_h

    for idx, (_, row) in enumerate(records.iterrows(), start=1):
        q = safe_get(row, ["問題文","設問","問題","本文"])

        choices = []
        for i in range(1, 6):
            v = safe_get(row, [f"選択肢{i}"])
            if v:
                choices.append((i, v))

        ans = safe_get(row, ["正解","正答","解答","答え"])
        cat = safe_get(row, ["科目分類","分類","科目"])
        code = safe_get(row, ["問題番号ID","問題番号"])

        # 画像の事前取得
        pil = None
        img_est_h = 0
        link_raw = safe_get(row, ["リンクURL","画像URL","画像リンク","リンク","URL"])
        if link_raw:
            try:
                image_url = convert_google_drive_link(link_raw)
                resp = requests.get(image_url, timeout=5)
                pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
                iw, ih = pil.size
                scale = min(usable_width / iw, page_usable_h / ih, 1.0)
                nw, nh = iw * scale, ih * scale
                img_est_h = nh + 20
            except Exception:
                pil = None
                img_est_h = len(wrapped_lines("", "[画像読み込み失敗]", usable_width, JAPANESE_FONT, 12)) * line_h

        # 高さ見積り
        est_h = 0
        q_lines = wrapped_lines("問題文: ", q, usable_width, JAPANESE_FONT, 12)
        est_h += len(q_lines) * line_h
        choice_lines_list = []
        for i, v in choices:
            ls = wrapped_lines(f"選択肢{i}: ", v, usable_width, JAPANESE_FONT, 12)
            choice_lines_list.append(ls)
            est_h += len(ls) * line_h
        est_h += img_est_h if img_est_h else 0
        ans_lines = wrapped_lines("正解: ", ans, usable_width, JAPANESE_FONT, 12)
        cat_lines = wrapped_lines("分類: ", cat, usable_width, JAPANESE_FONT, 12)
        code_lines = wrapped_lines("問題番号: ", code, usable_width, JAPANESE_FONT, 12)
        est_h += (len(ans_lines)+len(cat_lines)+len(code_lines)) * line_h + 20

        # ページ先頭を必ず問題文から
        if y - est_h < bottom_margin:
            new_page()

        # 描画
        draw_wrapped_lines(q_lines)
        for ls in choice_lines_list:
            draw_wrapped_lines(ls)

        if pil is not None:
            try:
                iw, ih = pil.size
                scale = min(usable_width / iw, page_usable_h / ih, 1.0)
                nw, nh = iw * scale, ih * scale
                if y - nh < bottom_margin:
                    new_page()
                remaining = y - bottom_margin
                if nh > remaining:
                    adj = remaining / nh
                    nw, nh = nw * adj, nh * adj
                img_io = io.BytesIO()
                pil.save(img_io, format="PNG")
                img_io.seek(0)
                img_reader = ImageReader(img_io)
                c.drawImage(img_reader, left_margin, y - nh, width=nw, height=nh, preserveAspectRatio=True, mask='auto')
                y -= nh + 20
            except Exception as e:
                err_lines = wrapped_lines("", f"[画像読み込み失敗: {e}]", usable_width, JAPANESE_FONT, 12)
                draw_wrapped_lines(err_lines)
        else:
            if link_raw:
                draw_wrapped_lines(wrapped_lines("", "[画像読み込み失敗]", usable_width, JAPANESE_FONT, 12))

        draw_wrapped_lines(ans_lines)
        draw_wrapped_lines(cat_lines)
        draw_wrapped_lines(code_lines)

        if y - 20 < bottom_margin:
            new_page()
        else:
            y -= 20

        if st.session_state.get("progress_on"):
            st.session_state["progress"].progress(min(idx / max(total, 1), 1.0))

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ===== PDF 生成 =====
if "pdf_bytes" not in st.session_state:
    st.session_state["pdf_bytes"] = None

if st.button("🖨️ PDFを作成（画像付き）"):
    st.session_state["progress_on"] = True
    st.session_state["progress"] = st.progress(0.0)
    with st.spinner("PDFを作成中…"):
        st.session_state["pdf_bytes"] = create_pdf(df_filtered)
    st.session_state["progress_on"] = False
    st.success("✅ PDF作成完了！")

if st.session_state["pdf_bytes"] is not None:
    st.download_button(
        label="📄 ヒット結果をPDFダウンロード",
        data=st.session_state["pdf_bytes"],
        file_name=f"{file_prefix}.pdf",
        mime="application/pdf"
    )

# ===== 画面の一覧（正解は初期非表示）=====
st.markdown("### 🔍 ヒットした問題一覧")
for i, (_, record) in enumerate(df_filtered.iterrows()):
    title = safe_get(record, ["問題文","設問","問題","本文"])
    with st.expander(f"{i+1}. {title[:50]}..."):
        st.markdown("### 📝 問題文")
        st.write(title)

        st.markdown("### ✏️ 選択肢")
        for j in range(1, 6):
            val = safe_get(record, [f"選択肢{j}"])
            if val:
                st.write(f"- {val}")

        show_ans = st.checkbox("正解を表示する", key=f"show_answer_{i}", value=False)
        if show_ans:
            st.markdown(f"**✅ 正解:** {safe_get(record, ['正解','正答','解答','答え'])}")
        else:
            st.markdown("**✅ 正解:** |||（クリックで表示）|||")

        st.markdown(f"**📚 分類:** {safe_get(record, ['科目分類','分類','科目'])}")
        code = safe_get(record, ["問題番号ID","問題番号"])
        if code:
            st.markdown(f"**🆔 問題番号:** {code}")

        link = safe_get(record, ["リンクURL","画像URL","画像リンク","リンク","URL"])
        if link:
            st.markdown(f"[画像リンクはこちら]({convert_google_drive_link(link)})")
        else:
            st.write("（画像リンクはありません）")