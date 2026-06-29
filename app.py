import streamlit as st
from openai import OpenAI
import json

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SYSTEM_PROMPT = """
You are an expert coaching mentor.
Classify the given coaching transcript line by line into one of these 5 categories based on the context.

Categories:
1. 指示: Giving orders, teaching what to do, or specifying concrete actions.
2. 提案: Presenting own opinions but leaving choices to the coachee (e.g., "How about...?").
3. 質問: Asking questions to the coachee.
4. 委譲: Encouraging the coachee to think, or waiting for their voluntary speech.
5. その他: Acknowledgment ("Yes", "I see"), mirroring, empathy, or any statements by the player.

Output MUST be a pure JSON array of objects, with NO markdown code blocks, NO extra text.
Format:
[
  {"text": "statement1", "label": "指示"},
  {"text": "statement2", "label": "提案"},
  {"text": "statement3", "label": "質問"},
  {"text": "statement4", "label": "委譲"},
  {"text": "statement5", "label": "その他"}
]
"""

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
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        return json.loads(content)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

COLOR_MAP = {
    "指示": {"bg": "#ffe6e6", "text": "#b30000"},
    "提案": {"bg": "#fff2e6", "text": "#cc6600"},
    "質問": {"bg": "#e6f2ff", "text": "#0066cc"},
    "委譲": {"bg": "#e6ffe6", "text": "#008000"},
    "その他": {"bg": "#f2f2f2", "text": "#333333"}
}

st.title("🎙️ コーチング逐語分析アプリ")
st.write("音声やテキストから発言の種類を分析し、後から手動で変更できます。")

if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None

tab1, tab2 = st.tabs(["🎵 音声ファイルから分析", "📝 テキストから直接分析"])

with tab1:
    uploaded_file = st.file_uploader("音声ファイルをアップロード", type=["mp3", "wav", "m4a", "mp4"])
    if uploaded_file is not None:
        if st.button("① 音声を文字起こしする"):
            with st.spinner("変換中..."):
                try:
                    audio_bytes = uploaded_file.read()
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=(uploaded_file.name, audio_bytes)
                    )
                    
                    sep_prompt = "話者が変わるごとに改行（1行1発言形式）にしてください。ラベルは不要です。"
                    sep_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": sep_prompt},
                            {"role": "user", "content": transcript.text}
                        ],
                        temperature=0.2
                    )
                    st.session_state['transcript_text'] = sep_response.choices[0].message.content
                    st.success("完了しました！")
                except Exception as e:
                    st.error(f"Error: {e}")

    if 'transcript_text' in st.session_state:
        edited_text = st.text_area("文字起こしされたテキスト", value=st.session_state['transcript_text'], height=200)
        if st.button("② このテキストを分析する"):
            with st.spinner("分析中..."):
                res = analyze_text(edited_text)
                if res:
                    st.session_state['analysis_results'] = res

with tab2:
    user_input = st.text_area("逐語録テキストを貼り付けてください", height=200)
    if st.button("テキストを分析する"):
        if user_input:
            with st.spinner("分析中..."):
                res = analyze_text(user_input)
                if res:
                    st.session_state['analysis_results'] = res

if st.session_state['analysis_results'] is not None:
    st.markdown("---")
    st.markdown("### 📊 分析結果と修正")
    
    st.markdown("""
    <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa;">
        <span style="background-color: #ffe6e6; color: #b30000; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 指示</span>
        <span style="background-color: #fff2e6; color: #cc6600; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 提案</span>
        <span style="background-color: #e6f2ff; color: #0066cc; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 質問</span>
        <span style="background-color: #e6ffe6; color: #008000; padding: 2px 6px; border-radius: 3px; margin-right: 10px;">■ 委譲</span>
        <span style="background-color: #f2f2f2; color: #333333; padding: 2px 6px; border-radius: 3px;">■ その他</span>
    </div>
    """, unsafe_allow_html=True)

    for idx, item in enumerate(st.session_state['analysis_results']):
        current_label = item['label']
        if current_label not in COLOR_MAP:
            current_label = "その他"
            
        style = COLOR_MAP[current_label]
        col1, col2 = st.columns([4, 1])
        
        with col1:
            html = f"""
            <div style="background-color: {style['bg']}; color: {style['text']}; padding: 8px; margin-bottom: 4px; border-radius: 4px;">
                <strong>【{current_label}】</strong> {item['text']}
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
            
        with col2:
            options = ["指示", "提案", "質問", "委譲", "その他"]
            try:
                default_idx = options.index(current_label)
            except ValueError:
                default_idx = 4
                
            new_label = st.selectbox(
                f"Change ({idx+1})", 
                options, 
                index=default_idx, 
                key=f"select_{idx}",
                label_visibility="collapsed"
            )
            
            if new_label != current_label:
                st.session_state['analysis_results'][idx]['label'] = new_label
                st.rerun()
