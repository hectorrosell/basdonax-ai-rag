
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def response_aina (query: str) -> str:
    # Parse the command line arguments
    ## args = parse_arguments()

    ## embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    ## db = Chroma(client=CHROMA_SETTINGS, embedding_function=embeddings)

    ## retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    ## callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    ## llm = Ollama(model=model, callbacks=callbacks, temperature=0, base_url='http://ollama:11434')

    ## prompt = assistant_prompt()

    ##def format_docs(docs):
    ##    return "\n\n".join(doc.page_content for doc in docs)

    ##rag_chain = (
    ##        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    ##        | prompt
    ##        | llm
    ##        | StrOutputParser()
    ##)

    ######### Run following module from the projecte-aina ###########

    # https://huggingface.co/projecte-aina/FLOR-6.3B

    ## --- console
    ## Gated model: Login with a HF token with gated access permission

    # MODULES

    # 	$	pip install transformers

    # 	$	pip install torch

    # 	$	pip install tensorflow

    # 	$	pip install flax

    # 	$	pip install 'accelerate>=0.26.0'

    #  Use a pipeline as a high-level helper

    # 	$	pip install git+https://github.com/huggingface/accelerate

    # 	$	huggingface-cli login

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
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generator = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        generation = generator(
            input_text,
            do_sample=True,
            top_k=10,
            eos_token_id=tokenizer.eos_token_id,
        )

        print(f"Result: {generation[0]['generated_text']}")

    return f"Result: {generation[0]['generated_text']}"

    ##return rag_chain.invoke(query)