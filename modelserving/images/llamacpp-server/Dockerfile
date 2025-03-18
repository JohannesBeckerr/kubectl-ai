FROM debian:latest AS llamacpp-builder

RUN apt-get update
RUN apt-get install -y g++ git cmake

WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp

WORKDIR /src/llama.cpp
RUN git checkout b4880

# -DCMAKE_EXE_LINKER_FLAGS="-static" should avoid libc dependency
# -DGGML_OPENMP=OFF avoids problem with linking gomp (TODO: is this a performance issue?)
RUN cmake -DCMAKE_EXE_LINKER_FLAGS="-static" -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=off .
RUN cmake --build . -j16 --config Release --target llama-server

#RUN ldd /src/llama.cpp/bin/llama-server

FROM gcr.io/distroless/static-debian12:latest

COPY --from=llamacpp-builder /src/llama.cpp/bin/llama-server /llama-server

ENTRYPOINT [ "/llama-server" ]
