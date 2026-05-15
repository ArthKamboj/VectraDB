# VectraDB - A semantic search vector database

A high-performance vector database built from scratch in modern C++.

VectraDB is designed for efficient similarity search on high-dimensional embeddings with a focus on speed, scalability, and low-level systems engineering. It combines custom indexing, optimized distance computation, durable storage, and a high-speed networking layer into a single lightweight engine.

---

##  Features

###  Similarity Search Engine
- Custom implementation of:
  - **L2 (Euclidean) Distance**
  - **Cosine Similarity**
- Optimized for fast nearest-neighbor retrieval on dense vector embeddings.

###  Intelligent Indexing
- **IVF (Inverted File Index)** powered by **K-Means Clustering**
- Reduces expensive brute-force `O(N)` searches.
- Efficient candidate pruning for scalable vector retrieval.

###  Thread-Safe Memory Management
- Custom read/write locking mechanism.
- Supports concurrent reads with safe write operations.
- Built for multi-threaded workloads and high query throughput.

###  Durable Storage Layer
- Custom binary **Write-Ahead Log (WAL)** implementation.
- Ensures persistence and crash recovery.
- Efficient append-only binary serialization.

###  High-Speed Networking
- gRPC + Protocol Buffers API.
- Low-latency communication for distributed applications.
- Language-agnostic client integration.

---

##  Architecture Overview

```text
                +------------------+
                |   Client Apps    |
                +--------+---------+
                         |
                    gRPC API
                         |
        +----------------+----------------+
        |                                 |
+-------v--------+               +--------v-------+
| Query Engine   |               | WAL Persistence|
+-------+--------+               +--------+-------+
        |                                 |
        v                                 v
+-------+---------------------------------+-------+
|               Vector Storage Engine             |
+----------------------+--------------------------+
                       |
                       v
             +---------+----------+
             | IVF K-Means Index  |
             +--------------------+
