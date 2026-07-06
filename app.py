import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    return tokenizer, model

tokenizer, model = load_model()

st.title("SmolLM Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("اكتب رسالتك..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    text = tokenizer.apply_chat_template(
        st.session_state.messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    assistant_reply = response.split("assistant")[-1].strip()

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )

    st.rerun()