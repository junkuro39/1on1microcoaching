import streamlit as st
from openai import OpenAI

# OpenAIクライアントの初期化（StreamlitのSecretsからAPIキーを取得）
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- 分析用プロンプトの定義 ---
SYSTEM_PROMPT = """
あなたはプロのコーチングメンターであり、1対1のコーチングにおける発話分析の専門家です。
クライアント（受講者）が提示したコーチングの逐語録（1行1発言のテキスト）を、以下の【分析ルール】と【出力フォーマット】に従って、厳密に色分け・分析してください。

【分析ルール】
入力された逐語録を1行ずつ読み込み、その発言が以下の4つのうちどれに該当するかを判断してください。
1. [指示・提案]：アドバイス、教える、指示する、自分の意見を提案する発言。
2. [質問]：相手に問いかける発言（オープンクエスチョン、クローズドクエスチョン問わず）。
3. [委譲]：相手に考えさせる、沈黙を促す、相手の自主的な発言を促す・待つ発言。
4. [それ以外]：相槌（「はい」「なるほど」）、オウム返し、共感、状況説明、世間話など。

【重要】
・コーチの発言のみを分析対象とし、プレイヤー（部下・クライアント）の発言はすべて「[それ以外]」として処理してください。
・もし文脈からどちらの発言か判別が難しい場合は、内容からコーチらしいアプローチ（問いかけやフィードバック）をしているものをコーチとみなしてください。

【出力フォーマット】
結果は以下のHTML形式のみで出力してください。余計な解説、前置き、まとめの文章は一切含めないでください。
必ず各発言を <div> タグで囲み、該当する種類に応じて以下のスタイル（背景色と文字色）を適用してください。

・[指示・提案] の場合：
<div style="background-color: #ffe6e6; color: #b30000; padding: 8px; margin-bottom: 4px; border-radius: 4px;"><strong>【指示・提案】</strong> 発言内容</div>

・[質問] の場合：
<div style="background-color: #e6f2ff; color: #0066cc; padding: 8px; margin-bottom: 4px; border-radius: 4px;"><strong>【質問】</strong> 発言内容</div>

・[委譲] の場合：
<div style="background-color: #e6ffe6; color: #008000; padding: 8px; margin-bottom: 4px; border-radius: 4px;"><strong>【委譲】</strong> 発言内容</div>

・[それ以外] の場合：
<div style="background-color: #f2f2f2; color: #333333; padding: 8px; margin-bottom: 4px; border-radius: 4px;">発言内容</div>
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
        return response.choices[0].message.content
    except Exception as e:
        return f"<div style='color:red;'>分析エラーが発生しました: {e}</div>"

# --- 結果表示関数の定義 ---
def display_results(html_content):
    st.markdown("### 📊 分析結果")
    st.markdown("""
    <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa;">
        <span style="background-color: #ffe6e6; color: #b30000; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 指示・提案</span>
        <span style="background-color: #e6f2ff; color: #0066cc; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 質問</span>
        <span style="background-color: #e6ffe6; color: #008000; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 委譲</span>
        <span style="background-color: #f2f2f2; color: #333333; padding: 2px 6px; border-radius: 3px;">■ 相槌・その他</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(html_content, unsafe_allow_html=True)

# --- 画面の構築 ---
st.title("🎙️ コーチング逐語分析アプリ")
st.write("音声ファイルのアップロード、またはテキスト入力から「指示・提案・質問・委譲」を色分け分析します。")

tab1, tab2 = st.tabs(["🎵 音声ファイルから分析", "📝 テキストから直接分析"])

# --- タブ1: 音声から文字起こし ---
with tab1:
    uploaded_file = st.file_uploader("ボタイムなどの音声ファイルをアップロード (mp3, wav, m4aなど) ", type=["mp3", "wav", "m4a", "mp4"])
    if uploaded_file is not None:
        if st.button("① 音声を文字起こしする"):
            with st.spinner("AIが音声からテキストに変換しています..."):
                try:
                    audio_bytes = uploaded_file.read()
                    
                    # 1. 音声から生のテキストを抽出
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=(uploaded_file.name, audio_bytes)
                    )
                    raw_text = transcript.text

                    # 2. GPTを使って改行・整形
                    separation_prompt = """
                    与えられた文章は、コーチとプレイヤーの1対1の会話を文字起こししたものです。
                    文脈から「コーチの発言」と「プレイヤーの発言」を見極め、話者が変わるごとに必ず改行（1行1発言の形式）にしてください。
                    余計な解説や「コーチ：」「プレイヤー：」などのラベルは一切付けず、純粋な発言内容だけのテキストにして出力してください。
                    """
                    
                    separation_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": separation_prompt},
                            {"role": "user", "content": raw_text}
                        ],
                        temperature=0.2
                    )
                    
                    st.session_state['transcript_text'] = separation_response.choices[0].message.content
                    st.success("話者ごとに自動改行して文字起こしが完了しました！")
                except Exception as e:
                    st.error(f"文字起こしエラー: {e}")

    # 文字起こし結果の編集エリアと分析ボタン（インデントを正確に配置）
    if 'transcript_text' in st.session_state:
        edited_text = st.text_area(
            "文字起こしされたテキスト (改行ごとに1発言として分析されます) ",
            value=st.session_state['transcript_text'],
            key="edited_transcript_text",
            height=300
        )
        if st.button("② このテキストを分析する"):
            res = analyze_text(edited_text)
            st.session_state['current_raw_results'] = res

# --- タブ2: テキストから直接分析 ---
with tab2:
    user_input = st.text_area("すでに文字起こしされたテキストをここに貼り付けてください (1行1発言) ", height=300)
    if st.button("テキストを分析する"):
        if user_input:
            res = analyze_text(user_input)
            st.session_state['current_raw_results'] = res
        else:
            st.warning("テキストを入力してください。")

# --- 分析画面の表示処理（一番左の壁にくっつけます） ---
if "current_raw_results" in st.session_state:
    display_results(st.session_state['current_raw_results'])
