import streamlit as st
import json
from openai import OpenAI

# ページの設定
st.set_page_config(page_title="コーチング発言分析アプリ", layout="centered")

st.title("🎙️ コーチング逐語分析アプリ")
st.write("音声ファイルのアップロード、またはテキスト入力から「指示・提案・質問・委譲」を色分け分析します。")

# 共通のAPIキーを設定（管理者側で設定、または利用者に最初に入力してもらう）
# ※セキュリティのため、本来は環境変数やStreamlitのSecrets機能を使うのが安全です。
# Secretsから自動的にAPIキーを読み込む、なければサイドバーから入力
if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
    API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("OpenAI API Keyを入力してください", type="password")

if not API_KEY:
    st.info("左側のサイドバーにOpenAIのAPIキーを入力すると、アプリが利用可能になります。")
else:
    client = OpenAI(api_key=API_KEY)

    # タブの作成（音声から文字起こしするか、直接テキストを貼るか）
    tab1, tab2 = st.tabs(["🎵 音声ファイルから分析", "📝 テキストから直接分析"])

    # 色の定義とCSSの埋め込み
    COLORS = {"指示": "#ffcccb", "提案": "#ffe4b5", "質問": "#e0ffff", "委譲": "#d3ffce", "その他": "#ffffff"}
    
    # 共通の分析用プロンプト
    system_prompt = """
あなたは優秀なコーチングアナリストです。与えられたコーチの発言を、以下の5つのいずれかに厳密に分類してください。

1. 指示 (Command / Tell): クライアントに次のアクションを促す発言。強い命令ではなく、「〜してください」「〜をやってみましょう」という日常的な促しで十分です。

2. 提案 (Suggestion / Sell): アイデアや選択肢を提示し、クライアントに働きかける発言。「〜をやってみますか？」「〜という方法もありますがどうですか？」というニュアンスです。

3. 質問 (Question): クライアントへの問いかけ、内省、思考、気づき、または状況の確認を促す発言。

4. 委譲 (Delegation): 決定権や今後の行動の主導権を全面的に相手に委ねる発言。「まかせるのでやってみてください」「あなたのやり方で進めてみてください」といったニュアンスです。

5. その他 (Other): 挨拶、解説・説明、相槌、セッションの進行管理など、上記のいずれにも該当しないもの。

出力は必ず以下のJSONフォーマットのみにしてください。
{"category": "分類名"}
"""

    def analyze_text(text_str):
        lines = [line.strip() for line in text_str.split('\n') if line.strip()]
        results = []
        progress_bar = st.progress(0)
        
        for i, line in enumerate(lines):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"発言: 「{line}」"}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                result = json.loads(response.choices[0].message.content)
                results.append((line, result.get("category", "その他")))
            except:
                results.append((line, "その他"))
            progress_bar.progress((i + 1) / len(lines))
        return results

    def display_results(analyzed_results):
        st.subheader("📊 分析結果 (手動で変更可能)")
    
        # 凡例表示
        cols = st.columns(5)
        for idx, (cat, col) in enumerate(COLORS.items()):
            html_legend = f'<div style="background-color:{col}; padding:5px; text-align:center; border-radius:4px; font-weight:bold; color:black;">{cat}</div>'
            cols[idx].markdown(html_legend, unsafe_allow_html=True)
    
        st.write("") # スペース
    
        # セッション内に「辞書型」の保存ポケットを確実に用意する
        if "editable_dict" not in st.session_state:
            st.session_state.editable_dict = {}
    
        # AIから送られてきたデータをループ処理
        for idx, (line, category) in enumerate(analyzed_results):
            c1, c2 = st.columns([1, 4])
            
            with c1:
                options = list(COLORS.keys())
                
                # 過去に手動変更していればそれを最優先し、なければAIの判定を使う
                if idx in st.session_state.editable_dict:
                    saved_cat = st.session_state.editable_dict[idx]
                else:
                    saved_cat = category
                    
                default_index = options.index(saved_cat) if saved_cat in options else 4
                
                # 手動変更用のプルダウン
                new_cat = st.selectbox(
                    f"分類選択_{idx}",
                    options,
                    index=default_index,
                    label_visibility="collapsed",
                    key=f"select_{idx}"
                )
                
                # ユーザーがプルダウンを切り替えた瞬間だけポケットを更新して再描画
                if new_cat != saved_cat:
                    st.session_state.editable_dict[idx] = new_cat
                    st.rerun()
    
            with c2:
                color = COLORS.get(new_cat, "#ffffff")
                html_content = f"""
                <div style="background-color:{color}; padding:10px; margin-bottom:8px; border-radius:4px; color:black; min-height:43px;">
                    {line}
                </div>
                """
                st.markdown(html_content, unsafe_allow_html=True)
    # --- タブ1: 音声から文字起こし ---
with tab1:
    uploaded_file = st.file_uploader("ボタイムなどの音声ファイルをアップロード (mp3, wav, m4aなど) ", type=["mp3", "wav", "m4a", "mp4"])
    if uploaded_file is not None:
        if st.button("① 音声を文字起こしする"):
            with st.spinner("AIが音声をテキストに変換しています..."):
                try:
                    # 音声のデータを安全に読み込みます
                    audio_bytes = uploaded_file.read()
                    
                    # OpenAI Whisper APIで文字起こし（※.nameの後に半角カンマを入れました）
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=(uploaded_file.name, audio_bytes)
                    )
                    st.session_state['transcript_text'] = transcript.text
                    st.success("文字起こしが完了しました！下のテキストを確認・編集してください。")
                except Exception as e:
                    st.error(f"文字起こしエラー: {e}")

           # 文字起こし結果の編集エリア
if 'transcript_text' in st.session_state:
    edited_text = st.text_area(
        "文字起こしされたテキスト (改行ごとに1発言として分析されます) ", 
        value=st.session_state['transcript_text'], 
        key="edited_transcript_text", # ←これでボタンを押してもデータが消えなくなります
        height=300
    )
    if st.button("② このテキストを分析する"):
        res = analyze_text(edited_text)
        display_results(res)

   # --- タブ2: テキストから直接分析 ---
with tab2:
    user_input = st.text_area("すでに文字起こしされたテキストをここに貼り付けてください（1行1発言）", height=300)
    if st.button("テキストを分析する"):
        if user_input:
            res = analyze_text(user_input)
            # 新しい分析結果が来たら、手動変更用の記憶ポケットをリセットして新しく保存する
            st.session_state.current_raw_results = res
            st.session_state.editable_dict = {}
        else:
            st.warning("テキストを入力してください。")

# --- 分析画面の表示処理（一番左の壁にくっつけます） ---
if "current_raw_results" in st.session_state:
    display_results(st.session_state.current_raw_results)
