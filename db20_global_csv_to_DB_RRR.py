import streamlit as st
import pandas as pd
import io
import csv
import requests
from PIL import Image
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.utils import ImageReader
from pathlib import Path
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import re

# =========================================================
# フォント設定（IPAex を優先、無ければCIDフォントへフォールバック）
# =========================================================
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

st.set_page_config(page_title="🔍 csv問題drop->検索DB_txt_pdf_goodnote-csv_csv出力", layout="wide")
st.title("🔍 csv問題drop->検索DB_txt_pdf_goodnote-csv_csv出力")

# =========================================================
# 文字・改行ユーティリティ
# =========================================================
def _strip(s):
    if s is None:
        return ""
    return str(s).replace("\ufeff", "").strip()

def _norm_space(s):
    # 列名用：空白・改行・タブを除去（データ本体には使わない）
    return re.sub(r"[\u3000 \t\r\n]+", "", _strip(s))

def _normalize_newlines(text: str, newline: str = "\n") -> str:
    if text is None:
        return ""
    t = re.sub(r"\r\n|\r", "\n", str(text))
    if newline == "\r\n":
        t = t.replace("\n", "\r\n")
    return t

def _safe_filename(s: str) -> str:
    s = _strip(s)
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:80] if s else "search"

# =========================================================
# 列名正規化
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 列名クレンジング（列名のみ）
    df.rename(columns={c: _norm_space(c) for c in df.columns}, inplace=True)

    # よくある別名 → 正式名
    alias = {
        "問題文":  ["設問", "問題", "本文"],
        "選択肢1": ["選択肢Ａ","選択肢a","A","ａ"],
        "選択肢2": ["選択肢Ｂ","選択肢b","B","ｂ"],
        "選択肢3": ["選択肢Ｃ","選択肢c","C","ｃ"],
        "選択肢4": ["選択肢Ｄ","選択肢d","D","ｄ"],
        "選択肢5": ["選択肢Ｅ","選択肢e","E","ｅ"],
        "正解":    ["解答","答え","ans","answer","正答"],
        "科目分類": ["分類","科目","カテゴリ","カテゴリー","分野"],
        "リンクURL": ["画像URL","画像リンク","リンク","画像Link","URL","url"],
        "問題番号ID": ["問題番号", "識別番号", "ID", "番号"],
    }
    colset = set(df.columns)
    for canon, cands in alias.items():
        if canon in colset:
            continue
        for c in cands:
            if c in colset:
                df.rename(columns={c: canon}, inplace=True)
                colset.add(canon)
                break

    return df

def safe_get(row: pd.Series | dict, keys, default=""):
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
            s = str(v) if v is not None else ""
            s = s.strip()
            if s != "":
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

# =========================================================
# ★ 文字コードを「日本語として自然に読めるもの」に寄せるデコード
# =========================================================
def _decode_best_effort(raw: bytes) -> str:
    candidates = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc_jp"]

    for enc in candidates:
        try:
            return raw.decode(enc)
        except Exception:
            pass

    best_text = None
    best_score = None
    for enc in ["cp932", "shift_jis", "euc_jp", "utf-8-sig", "utf-8"]:
        try:
            t = raw.decode(enc, errors="replace")
        except Exception:
            continue

        rep = t.count("\ufffd")
        jp = sum(
            1 for ch in t
            if ("\u3040" <= ch <= "\u30ff") or ("\u4e00" <= ch <= "\u9fff")
        )
        score = rep * 1000 - jp
        if best_score is None or score < best_score:
            best_score = score
            best_text = t

    if best_text is not None:
        return best_text

    return raw.decode("utf-8", errors="ignore")

# =========================================================
# ★ ヘッダ有無判定＆自動ヘッダ付与（←③対策の本体）
# =========================================================
def _looks_like_header(cols: list[str]) -> bool:
    joined = "".join(cols)
    header_tokens = ["問題文", "設問", "選択肢", "正解", "解答", "分類", "科目", "リンク", "URL", "ID", "番号"]
    return any(tok in joined for tok in header_tokens)

def _default_header_by_ncol(ncol: int) -> list[str]:
    # まず「問題文 + 選択肢1-5」は共通として寄せる
    base = ["問題文", "選択肢1", "選択肢2", "選択肢3", "選択肢4", "選択肢5"]

    # ③は ncol=8 で「正解(空)」「科目分類」
    if ncol == 8:
        return base + ["正解", "科目分類"]

    # よくある想定：+ 正解 + 科目分類 + 問題番号ID + リンクURL（計10）
    if ncol == 10:
        return base + ["正解", "科目分類", "問題番号ID", "リンクURL"]

    # 9列：リンク無し or 問題番号無し等のケース
    if ncol == 9:
        # 「正解・科目分類・問題番号ID」までを優先
        return base + ["正解", "科目分類", "問題番号ID"]

    # 7列：正解無しで分類まで など
    if ncol == 7:
        return base + ["科目分類"]

    # それ以外：とりあえずcol_1..で作る（壊れないこと優先）
    return [f"col{i+1}" for i in range(ncol)]

# =========================================================
# ★ CSV読み込み（ヘッダ補正＋列数補正＋ヘッダ無し対応）
# =========================================================
def read_csv_safely_with_column_fix(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = _decode_best_effort(raw)

    lines = text.splitlines()
    if not lines:
        return pd.DataFrame()

    # 1行目の区切り文字ゆらぎ対策（ヘッダ候補のみ）
    first_line = lines[0].replace("、", ",")
    first_cols = next(csv.reader([first_line]))

    has_header = _looks_like_header([c.strip() for c in first_cols])

    if has_header:
        header = [h.strip().replace("\ufeff", "") for h in first_cols]
        data_lines = lines[1:]
    else:
        # ★ヘッダ無し：1行目はデータ
        ncol = len(first_cols)
        header = _default_header_by_ncol(ncol)
        data_lines = lines  # 1行目から全部データ

    ncol = len(header)

    fixed_rows = []
    reader = csv.reader(data_lines)
    for row in reader:
        if not row or all((c.strip() == "" for c in row)):
            continue

        # 列が多すぎる：余りを先頭列へ吸収
        while len(row) > ncol:
            row[0] = row[0] + "," + row[1]
            del row[1]

        # 列が足りない：右を空で埋める
        if len(row) < ncol:
            row = row + [""] * (ncol - len(row))

        fixed_rows.append(row)

    df = pd.DataFrame(fixed_rows, columns=header).fillna("")
    return df

# =========================================================
# 入力CSV（Drag & Drop）
# =========================================================
uploaded = st.file_uploader("CSVをドラッグ＆ドロップ（または選択）", type=["csv"])
if uploaded is None:
    st.info("任意のCSVをアップロードしてください（ヘッダあり推奨）。")
    st.stop()

df = read_csv_safely_with_column_fix(uploaded)
df = normalize_columns(df)

# URL列だけのCSVでも「リンクURL」に寄せる
if "リンクURL" not in df.columns and "URL" in df.columns:
    df.rename(columns={"URL": "リンクURL"}, inplace=True)

# =========================================================
# 検索
# =========================================================
query = st.text_input("検索語（例：どれか / エナメル質 / 1459）")
st.caption("💡 `&` でAND検索（例: レジン & 理工）")

if not query:
    st.stop()

keywords = [kw.strip() for kw in query.split("&") if kw.strip()]

def row_text(r: pd.Series) -> str:
    vals = []
    for v in r.values:
        if v is None:
            continue
        s = str(v)
        if s.strip() == "":
            continue
        vals.append(s)
    return " ".join(vals)

def match_all_keywords(row: pd.Series) -> bool:
    text = row_text(row).casefold()
    return all(kw.casefold() in text for kw in keywords)

df_filtered = df[df.apply(match_all_keywords, axis=1)].reset_index(drop=True)
st.info(f"{len(df_filtered)}件ヒットしました")

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
file_prefix = f"{_safe_filename(query)}_{timestamp}"

# =========================================================
# CSV ダウンロード
# =========================================================
csv_buffer = io.StringIO()
ensure_output_columns(df_filtered).to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 ヒット結果をCSVダウンロード",
    data=csv_buffer.getvalue(),
    file_name=f"{file_prefix}.csv",
    mime="text/csv"
)

# =========================================================
# GoodNotes 用 CSV（Front/Back）
# =========================================================
def _gn_clean(s: str) -> str:
    return _strip(s).replace("　", "")

def _gn_make_front_back(row: pd.Series,
                        numbering: str = "ABC",
                        add_labels: bool = True,
                        add_meta: bool = True) -> tuple[str, str]:
    q = _gn_clean(safe_get(row, ["問題文"]))
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

    ans = _gn_clean(safe_get(row, ["正解"]))
    back = f"正解: {ans}" if add_labels else ans

    if add_meta:
        cat = _gn_clean(safe_get(row, ["科目分類"]))
        code = _gn_clean(safe_get(row, ["問題番号ID"]))
        extra = "\n".join([s for s in (cat, code) if s])
        if extra:
            back = back + "\n\n" + _normalize_newlines(extra)

    back = _normalize_newlines(back)
    return front, back

def dataframe_to_goodnotes_bytes(df_in: pd.DataFrame) -> bytes:
    base = ensure_output_columns(df_in)
    fronts, backs = [], []
    for _, row in base.iterrows():
        f, b = _gn_make_front_back(row)
        fronts.append(f); backs.append(b)
    out = pd.DataFrame({"Front": fronts, "Back": backs})
    for c in out.columns:
        out[c] = out[c].map(lambda v: _normalize_newlines(v, "\n"))
    buf = io.StringIO()
    buf.write("\ufeff")
    out.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")

st.download_button(
    label="📥 GoodNotes用CSV（Front/Back）をダウンロード",
    data=dataframe_to_goodnotes_bytes(df_filtered),
    file_name=f"{file_prefix}_goodnotes.csv",
    mime="text/csv",
)

# =========================================================
# TXT / PDF 共通
# =========================================================
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
    q = safe_get(row, ["問題文"])
    parts = [f"問題文: {q}"]
    for i in range(1, 6):
        choice = safe_get(row, [f"選択肢{i}"])
        if choice:
            parts.append(f"選択肢{i}: {choice}")
    parts.append(f"正解: {safe_get(row, ['正解'])}")
    parts.append(f"分類: {safe_get(row, ['科目分類'])}")
    code = safe_get(row, ["問題番号ID"])
    if code:
        parts.append(f"問題番号: {code}")
    link = safe_get(row, ["リンクURL"])
    if link:
        parts.append(f"画像リンク: {convert_google_drive_link(link)}（PDFに画像表示）")
    return "\n".join(parts)

# =========================================================
# TXT ダウンロード
# =========================================================
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

# =========================================================
# PDF 作成
# =========================================================
def create_pdf(records: pd.DataFrame):
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

    for _, row in records.iterrows():
        q = safe_get(row, ["問題文"])

        choices = []
        for i in range(1, 6):
            v = safe_get(row, [f"選択肢{i}"])
            if v:
                choices.append((i, v))

        ans = safe_get(row, ["正解"])
        cat = safe_get(row, ["科目分類"])
        code = safe_get(row, ["問題番号ID"])

        pil = None
        img_est_h = 0
        link_raw = safe_get(row, ["リンクURL"])
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
        est_h += (len(ans_lines) + len(cat_lines) + len(code_lines)) * line_h + 20

        if y - est_h < bottom_margin:
            new_page()

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
                c.drawImage(img_reader, left_margin, y - nh, width=nw, height=nh,
                            preserveAspectRatio=True, mask='auto')
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

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

if "pdf_bytes" not in st.session_state:
    st.session_state["pdf_bytes"] = None

if st.button("🖨️ PDFを作成（画像付き）"):
    with st.spinner("PDFを作成中…"):
        st.session_state["pdf_bytes"] = create_pdf(df_filtered)
    st.success("✅ PDF作成完了！")

if st.session_state["pdf_bytes"] is not None:
    st.download_button(
        label="📄 ヒット結果をPDFダウンロード",
        data=st.session_state["pdf_bytes"],
        file_name=f"{file_prefix}.pdf",
        mime="application/pdf"
    )

# =========================================================
# 画面の一覧（問題文末尾の識別番号を切らない）
# =========================================================
st.markdown("### 🔍 ヒットした問題一覧")
for i, (_, record) in enumerate(df_filtered.iterrows()):
    title = safe_get(record, ["問題文"])
    with st.expander(f"{i+1}. {title}"):
        st.markdown("### 📝 問題文")
        st.write(title)

        st.markdown("### ✏️ 選択肢")
        for j in range(1, 6):
            val = safe_get(record, [f"選択肢{j}"])
            if val:
                st.write(f"- {val}")

        show_ans = st.checkbox("正解を表示する", key=f"show_answer_{i}", value=False)
        if show_ans:
            st.markdown(f"**✅ 正解:** {safe_get(record, ['正解'])}")
        else:
            st.markdown("**✅ 正解:** |||（クリックで表示）|||")

        st.markdown(f"**📚 分類:** {safe_get(record, ['科目分類'])}")

        code = safe_get(record, ["問題番号ID"])
        if code:
            st.markdown(f"**🆔 問題番号:** {code}")

        link = safe_get(record, ["リンクURL"])
        if link:
            st.markdown(f"[画像リンクはこちら]({convert_google_drive_link(link)})")
        else:
            st.write("（画像リンクはありません）")
