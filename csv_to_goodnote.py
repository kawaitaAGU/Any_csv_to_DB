# app.py
import io
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="GoodNotes 単語帳CSV 変換", layout="wide")
st.title("📚 GoodNotes 単語帳CSV 変換（問題＋選択肢 → Front / 正解 → Back）")

st.caption(
    "入力CSVの列順は「問題文,選択肢1,選択肢2,選択肢3,選択肢4,選択肢5,正解,科目分類,リンクURL」を想定。"
    "列名にBOMや全角空白が混じっていても自動で正規化します。"
)

# =========================
# ユーティリティ
# =========================
REQUIRED = ["問題文","選択肢1","選択肢2","選択肢3","選択肢4","選択肢5","正解","科目分類","リンクURL"]

ALIAS = {
    "問題文":  ["設問", "問題", "本文"],
    "選択肢1": ["選択肢Ａ","選択肢a","A","ａ"],
    "選択肢2": ["選択肢Ｂ","選択肢b","B","ｂ"],
    "選択肢3": ["選択肢Ｃ","選択肢c","C","ｃ"],
    "選択肢4": ["選択肢Ｄ","選択肢d","D","ｄ"],
    "選択肢5": ["選択肢Ｅ","選択肢e","E","ｅ"],
    "正解":    ["解答","答え","ans","answer"],
    "科目分類": ["分類","科目","カテゴリ","カテゴリー"],
    "リンクURL": ["画像URL","画像リンク","リンク","画像Link"],
}

def _clean(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("\ufeff", "").strip().replace("　", "")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean(c) for c in df.columns]
    colset = set(df.columns)
    for canon, cands in ALIAS.items():
        if canon in colset:
            continue
        for c in cands:
            if c in colset:
                df.rename(columns={c: canon}, inplace=True)
                colset.add(canon)
                break
    # 足りない列は空で追加
    for c in REQUIRED:
        if c not in df.columns:
            df[c] = ""
    # 列順を合わせる
    return df[REQUIRED]

def make_front_back(row: pd.Series, numbering: str, add_labels: bool, add_meta: bool) -> Tuple[str, str]:
    """
    numbering: 'ABC' or '123'
    add_labels: Trueなら '正解: ' をBackにつける
    add_meta: Trueなら Backの末尾に 科目分類/リンクURL を付与
    """
    q = _clean(row.get("問題文", ""))
    choices = [
        _clean(row.get("選択肢1", "")),
        _clean(row.get("選択肢2", "")),
        _clean(row.get("選択肢3", "")),
        _clean(row.get("選択肢4", "")),
        _clean(row.get("選択肢5", "")),
    ]
    # 表示ラベル
    if numbering == "ABC":
        labels = ["A", "B", "C", "D", "E"]
    else:
        labels = ["1", "2", "3", "4", "5"]

    choice_lines = [f"{labels[i]}. {txt}" for i, txt in enumerate(choices) if txt]

    front = q if not choice_lines else q + "\n\n" + "\n".join(choice_lines)

    ans = _clean(row.get("正解", ""))
    back = f"正解: {ans}" if add_labels else ans

    if add_meta:
        subject = _clean(row.get("科目分類", ""))
        link = _clean(row.get("リンクURL", ""))
        extra = "\n".join([s for s in [subject, link] if s])
        if extra:
            back = back + "\n\n" + extra

    return front, back

def convert_df(df: pd.DataFrame, numbering: str, add_labels: bool, add_meta: bool) -> pd.DataFrame:
    fronts, backs = [], []
    for _, row in df.iterrows():
        f, b = make_front_back(row, numbering, add_labels, add_meta)
        fronts.append(f); backs.append(b)
    out = pd.DataFrame({"Front": fronts, "Back": backs})
    return out

# =========================
# オプション
# =========================
with st.sidebar:
    st.header("⚙️ オプション")
    numbering = st.radio("選択肢の番号", ["ABC", "123"], index=0, horizontal=True)
    add_labels = st.checkbox("Backの先頭に「正解: 」を付ける", value=True)
    add_meta = st.checkbox("Backに 科目分類/リンクURL を追記", value=False)
    st.caption("※追記は改行して末尾に追加します。")

# =========================
# 入力（Drag & Drop）
# =========================
uploaded = st.file_uploader("ここにCSVをドラッグ＆ドロップ（または選択）", type=["csv"])

if uploaded is None:
    st.info("サンプル形式：問題文,選択肢1,選択肢2,選択肢3,選択肢4,選択肢5,正解,科目分類,リンクURL")
    st.stop()

# pandasで読む（BOMや日本語もOK）
try:
    df_raw = pd.read_csv(uploaded, dtype=str).fillna("")
except Exception as e:
    st.error(f"CSVの読み込みに失敗しました: {e}")
    st.stop()

df = normalize_columns(df_raw)

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"必要な列が見つかりません: {missing}")
    st.stop()

st.success(f"✅ 読み込みOK：{len(df)}件")
with st.expander("📄 入力の先頭5行（正規化後）を確認", expanded=False):
    st.dataframe(df.head(5), use_container_width=True)

# 変換
out_df = convert_df(df, numbering=numbering, add_labels=add_labels, add_meta=add_meta)

with st.expander("👀 出力プレビュー（先頭10行）", expanded=True):
    st.dataframe(out_df.head(10), use_container_width=True)

# ダウンロード
buf = io.StringIO()
out_df.to_csv(buf, index=False, encoding="utf-8-sig")
csv_bytes = buf.getvalue().encode("utf-8-sig")

st.download_button(
    label="📥 変換CSVをダウンロード（Front,Back）",
    data=csv_bytes,
    file_name="goodnotes_flashcards.csv",
    mime="text/csv",
)

st.caption("出力はUTF-8(BOM付き)。Excelでも文字化けしにくい形式です。GoodNotesのインポートでご利用ください。")