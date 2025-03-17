FROM llamacpp-server:latest

# TODO: Add checksum
ADD https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf?download=true /gemma-3-12b-it-Q4_K_M.gguf

# Default model
ENV LLAMA_ARG_MODEL=/gemma-3-12b-it-Q4_K_M.gguf

# Bigger context size (though not too large given memory)
ENV LLAMA_ARG_CTX_SIZE=16384

ENTRYPOINT [ "/llama-server" ]
