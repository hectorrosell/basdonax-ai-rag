import streamlit as st

st.set_page_config(layout='wide', page_title='Aina', page_icon='⌨️')

from common.langchain_module import response
from common.streamlit_style import hide_streamlit_style
from common.aina_modules import response_aina

hide_streamlit_style()

from huggingface_hub import login

# fine
# login(token="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")

# write
login(token="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")

# Título de la aplicación Streamlit
st.title("projecte-aina/FLOR-6.3B - v3 - 09/10/2024")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Escribí tu mensaje 😎"):
    # Display user message in chat message container
    with st.chat_message("aina_user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "aina_user", "content": user_input})

if user_input != None:
    if st.session_state.messages and user_input.strip() != "":
        ## response = response(user_input)

        ## response_aina = response_aina(user_input)

        import os

        is_transformer = False

        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        if is_transformer:

            # Use a pipeline as a high-level helper
            from transformers import pipeline

            pipe = pipeline("text-generation", model="projecte-aina/FLOR-6.3B")

            # Load model directly
            from transformers import AutoTokenizer, AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained("projecte-aina/FLOR-6.3B")
            model = AutoModelForCausalLM.from_pretrained("projecte-aina/FLOR-6.3B")

        else:

            # With no transformers version

            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

            input_text = "Sovint em trobo pensant en tot allò que"

            model_id = "projecte-aina/FLOR-6.3B"

            tokenizer = AutoTokenizer.from_pretrained(model_id )

            print(f" Result: pre-generate text ... ")

            generator = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                max_new_tokens=20,
                clean_up_tokenization_spaces=True
            )

            print(f" TEST: post-generate text ... ")

            generation = generator(
                input_text,
                do_sample=True,
                top_k=10,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Display assistant response in chat message container
            with st.chat_message("aina_assistant"):
                st.markdown(f"Result: {generation[0]['generated_text']}")
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "aina_assistant", "content": f"Result: {generation[0]['generated_text']}"})
