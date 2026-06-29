import streamlit as st
from openai import OpenAI
import json

# OpenAIクライアントの初期化
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- 分析用プロンプトの定義（指示と提案を厳密に分離、JSONで出力させる） ---
SYSTEM_PROMPT = """
あなたはプロのコーチングメンターであり、1対1のコーチングにおける発話分析の専門家です。
クライアントが提示したコーチングの逐語録（1行1発言のテキスト）を、以下の【分析ルール】に従って、1行ずつ厳密に分類してください。

【分析ルール】
入力された逐語録を1行ずつ読み込み、その発言が以下の5つのうちどれに該当するかを判断してください。
1. 指示：命令する、こうしなさいと教える、具体的な行動を指定する発言。
2. 提案：自分の意見を伝える、「〜してみてはどうですか？」「〜という方法もあります」など、相手に選択の余地を残した発言。
3. 質問：相手に問いかける発言（オープンクエスチョン、クローズドクエスチョン問わず）。
4. 委譲：相手に考えさせる、沈黙を促す、相手の自主的な発言を促す・待つ発言。
5. その他：相槌（「はい」「なるほど」）、オウム返し、共感、状況説明、世間話、およびプレイヤー（部下）の発言すべて。

【重要】
・コーチの発言のみを分析対象とし、プレイヤー（部下・クライアント）の発言はすべて「その他」として処理してください。

【出力フォーマット】
必ず、以下の構造を持つJSON配列のみを出力してください。余計な解説や前置き、```json のようなマークダウンの枠も一切含めないでください。純粋なJSON文字列だけを返してください。

[
  {"text": "発言内容1", "label": "指示"},
  {"text": "発言内容2", "label": "提案"},
  {"text": "発言内容3", "label": "質問"},
  {"text": "発言内容4", "label": "委譲"},
  {"text": "発言内容5", "label": "その他"}
]
"""

# --- 分析関数の定義 ---
def analyze_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        # 万が一マークダウンの枠がついてしまった場合の除去
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        return json.loads(content)
    except Exception as e:
        st.error(f"分析エラーが発生しました: {e}")
        return None

# --- スタイルの定義 ---
COLOR_MAP = {
    "指示": {"bg": "#ffe6e6", "text": "#b30000", "label": "■ 指示"},
    "提案": {"bg": "#fff2e6", "text": "#cc6600", "label": "■ 提案"},
    "質問": {"bg": "#e6f2ff", "text": "#0066cc", "label": "■ 質問"},
    "委譲": {"bg": "#e6ffe6", "text": "#008000", "label": "■ 委譲"},
    "その他": {"bg": "#f2f2f2", "text": "#333333", "label": "■ 相槌・その他"}
}

# --- 画面の構築 ---
st.title("🎙️ コーチング逐語分析アプリ")
st.write("音声ファイルやテキストから「指示・提案・質問・委譲」を色分け分析し、後から手動で修正できます。")

# 記憶ポケットの初期化
if 'analysis_results' not_in st.session_state:
    st.session_state['analysis_results'] = None

tab1, tab2 = st.tabs(["🎵 音声ファイルから分析", "📝 テキストから直接分析"])

# --- タブ1: 音声から文字起こし ---
with tab1:
    uploaded_file = st.file_uploader("ボタイムなどの音声ファイルをアップロード", type=["mp3", "wav", "m4a", "mp4"])
    if uploaded_file is not None:
        if st.button("① 音声を文字起こしする"):
            with st.spinner("AIが音声からテキストに変換しています..."):
                try:
                    audio_bytes = uploaded_file.read()
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=(uploaded_file.name, audio_bytes)
                    )
                    
                    separation_prompt = "会話を文脈から見極め、話者が変わるごとに改行（1行1発言形式）にしてください。ラベルは不要です。"
                    separation_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": separation_prompt},
                            {"role": "user", "content": transcript.text}
                        ],
                        temperature=0.2
                    )
                    st.session_state['transcript_text'] = separation_response.choices[0].message.content
                    st.success("文字起こしが完了しました！")
                except Exception as e:
                    st.error(f"文字起こしエラー: {e}")

    if 'transcript_text' in st.session_state:
        edited_text = st.text_area("文字起こしされたテキスト", value=st.session_state['transcript_text'], height=200)
        if st.button("② このテキストを分析する"):
            with st.spinner("AIが発言を分類しています..."):
                res = analyze_text(edited_text)
                if res:
                    st.session_state['analysis_results'] = res

# --- タブ2: テキストから直接分析 ---
with tab2:
    user_input = st.text_area("すでに文字起こしされたテキストを貼り付けてください", height=200)
    if st.button("テキストを分析する"):
        if user_input:
            with st.spinner("AIが発言を分類しています..."):
                res = analyze_text(user_input)
                if res:
                    st.session_state['analysis_results'] = res
        else:
            st.warning("テキストを入力してください。")

# --- 📊 分析結果と手動修正エリア（一番左の壁） ---
if st.session_state['analysis_results'] is not None:
    st.markdown("---")
    st.markdown("### 📊 分析結果と修正")
    
    # 凡例表示
    st.markdown("""
    <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa;">
        <span style="background-color: #ffe6e6; color: #b30000; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 指示</span>
        <span style="background-color: #fff2e6; color: #cc6600; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 提案</span>
        <span style="background-color: #e6f2ff; color: #0066cc; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 質問</span>
        <span style="background-color: #e6ffe6; color: #008000; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 委譲</span>
        <span style="background-color: #f2f2f2; color: #333333; padding: 2px 6px; border-radius: 3px;">■ その他</span>
    </div>
    """, unsafe_allow_html=True)

    # 1行ずつループして、色付き表示 ＋ 横に修正用セレクトボックスを配置
    for idx, item in enumerate(st.session_state['analysis_results']):
        current_label = item['label']
        if current_label not_in COLOR_MAP:
            current_label = "その他"
            
        style = COLOR_MAP[current_label]
        
        # 画面を左（結果表示）と右（修正ボックス）に分割
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # 色付き発言の表示
            html = f"""
            <div style="background-color: {style['bg']}; color: {style['text']}; padding: 8px; margin-bottom: 4px; border-radius: 4px;">
                <strong>【{current_label}】</strong> {item['text']}
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
            
        with col2:
            # 手動修正用の選択肢
            options = ["指示", "提案", "質問", "委譲", "その他"]
            try:
                default_idx = options.index(current_label)
            except ValueError:
                default_idx = 4
                
            new_label = st.selectbox(
                f"変更 ({idx+1})", 
                options, 
                index=default_idx, 
                key=f"select_{idx}",
                label_visibility="collapsed"
            )
            
            # もしユーザーが選択を変えたら、記憶を上書きして画面を再描画
            if new_label != current_label:
                st.session_state['analysis_results'][idx]['label'] = new_label
                st.rerun()
