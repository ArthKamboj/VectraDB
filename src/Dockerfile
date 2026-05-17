FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    g++ \
    pkg-config \
    protobuf-compiler \
    protobuf-compiler-grpc \
    libgrpc++-dev \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server.cpp core_math.cpp vectordb.proto ./

RUN protoc -I=. --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) vectordb.proto

RUN g++ -std=c++17 -O3 -mavx2 server.cpp vectordb.pb.cc vectordb.grpc.pb.cc -o vectordb_server $(pkg-config --cflags --libs grpc++ protobuf)

EXPOSE 50051

CMD ["./vectordb_server"]