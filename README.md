# VectraDB AI
### **An Enterprise-Grade, Hardware-Accelerated Multi-Tenant Vector Search Engine**


VectraDB AI is a high-performance, containerized microservice architecture built from scratch. It ingests raw images, extracts their 512-dimensional semantic mathematics using OpenAI's CLIP model, and stores them in a custom C++ hardware-accelerated database for sub-millisecond semantic search.

Designed as a SaaS backend, it features logical multi-tenancy, custom disk persistence, and a highly optimized Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor (ANN) retrieval.

---

## System Architecture

The system is deployed as a 3-tier microservice cluster securely networked via Docker bridge.

```plaintext
=======================================================================
                         CLIENT TIER
=======================================================================

                      [ User A / User B ]
                               |
                               | Uploads Image & Queries
                               v
                    +-----------------------+
                    |  Streamlit Frontend   | (Web UI)
                    +-----------------------+

=======================================================================
                         API TIER (Python)
=======================================================================
                               |
                               | HTTP POST (JSON + File)
                               v
        +===================================================+
        |             API Gateway Container                 |
        |                                                   |
        |  +----------------+      +---------------------+  |
        |  | FastAPI Router |<---->| HuggingFace CLIP    |  |
        |  +--------+-------+      +---------------------+  |
        |           |                       |               |
        |           v                       | 512D Vector   |
        |  [ Shared Docker Volume ]---------+               |
        |  [ (Saves user_id.jpg)  ]                         |
        +===========+=======================================+

=======================================================================
                         DATABASE TIER (C++)
=======================================================================
                               |
                               | gRPC / Protocol Buffers (Binary)
                               v
        +===================================================+
        |            VectraDB Engine Container              |
        |                                                   |
        |  +-------------------------+                      |
        |  | Multi-Tenant Router     | (Checks user_id)     |
        |  +--------+----------------+                      |
        |           |                                       |
        |           v                                       |
        |  +-------------------------+                      |
        |  | AVX-256 CPU Vector Math | (Distance Calc)      |
        |  +--------+----------------+                      |
        |           |                                       |
        |           v                                       |
        |  +-------------------------+    +--------------+  |
        |  | HNSW Graph Memory       |--->| Custom WAL & |  |
        |  +-------------------------+    | Snapshots    |  |
        +=================================+--------------+==+
```

## Core Engineering Patterns

### 1. Hardware-Accelerated Vector Math (AVX-256)
Instead of relying on standard loops, the core math engine is written in C++ utilizing raw **Advanced Vector Extensions (AVX)** CPU lanes. It processes 8 floating-point numbers simultaneously, drastically reducing the CPU cycles required to calculate massive multi-dimensional Euclidean distances.

### 2. Logical Multi-Tenancy (SaaS Architecture)
The database utilizes a Tagged Memory Pattern. Every document, vector, and Write-Ahead Log (WAL) entry is securely bound to a `user_id`. The C++ engine enforces an absolute isolation wall during graph traversal, allowing hundreds of users to share the same hardware instance with zero risk of data leakage.

### 3. Custom Disk Persistence Engine
Instead of relying on standard SQL tables, VectraDB implements a ground-up storage engine featuring:
  * **Write-Ahead Logging (WAL):** O(1) disk writes ensuring zero data loss if the Docker container crashes.
  * **Binary Snapshotting:** Memory compaction algorithm that serializes the active HNSW graph into a dense `.bin` file, allowing near-instantaneous boot times and automatic WAL purging.

### 4. High Speed Protocol Buffers (gRPC)
The Python Gateway and C++ Engine communicate exclusively via gRPC. By serializing data into strongly-typed binary streams instead of JSON, the architecture achieves ultra-low latency internal networking, even when transmitting massive 512D float arrays.

## Technology Stack
| Component | Technology | Purpose |
|---|---|---|
| **Frontend UI** | Streamlit, Pillow (PIL) | Responsive Web Dashboard, Image rendering. |
| **API Gateway** | Python, FastAPI, Uvicorn | Async HTTP routing, file handling, and model serving. |
| **AI Vision Model** | SentenceTransformers | Hosts `clip-ViT-B-32` to map images to mathematical concepts. |
| **Network Protocol** | gRPC, Protocol Buffers | Binary serialization pipeline for inter-container communication. |
| **Database Engine** | C++17, AVX Intrinsics | Hardware-accelerated memory graph, HNSW index, Search logic. |
| **Storage Engine** | C++ Standard FStream | Custom Write-Ahead Log (WAL) and Binary Memory Snapshots. |
| **Infrastructure** | Docker, Docker Compose | Environment-agnostic containerization and virtual networking. |

## Local Deployment Guide

### Prerequisites
  * Docker and Docker Compose installed
  * Ports `8501`, `8000`, `8501` available on your local machine.

## Installation and Boot
  1. **Clone the Repository:**
     
     ```bash
     git clone https://github.com/ArthKamboj/VectraDB.git
     cd vectradb-ai
     ```
  2. **Build and boot the cluster in Detached Mode:**
     
     ```bash
     docker compose up --build -d
     ```
  3. **Monitor the Boot Sequence:**
     
     The API Gateway requires ~30 seconds to download the CLIP model into memory.
     ```bash
     docker compose logs -f api-gateway
     ```
  4. **Access the DashBoard**
     
     Open your browser and navigate to `http://localhost:8501`.

## Testing the Multi-Tenant Isolation

 **To verify the SaaS security architecture:**

  1. Open the UI, enter user_A as the username, and upload an image of a Car.

  2. Change the username to user_B, and upload an image of a Cat.

  3. While logged in as user_B, run a Vector Search.

Result: The C++ engine will securely isolate the graph, returning only the Cat imagery and completely walling off User A's data.

## Infrastructure Management

To gracefully spin down the microservices and virtual network:

  ```bash
  docker compose down
```
