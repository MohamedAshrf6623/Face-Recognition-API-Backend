# 🚀 Face Recognition API: Unified Service Architecture (Technical Documentation)

This repository holds the final, production-ready **Backend API** code for the Incremental Face Recognition System. The architecture employs a **Unified Service Pattern (MVC/Service)** within a single Flask application, guaranteeing highly scalable performance and maintainability on cloud platforms (AWS).

## 1. 🌟 Project Overview and Architecture

The system achieves stable prediction performance by consolidating services into a single application running on one port.

| Feature | Value | Technical Rationale |
| :--- | :--- | :--- |
| **Architecture** | **Unified Service Pattern (MVC)** | Enforces clean separation: logic in `services.py`, network traffic in `controllers.py`. This is crucial for stability and maintenance. |
| **Core AI** | FaceNet, SVM, MTCNN | Leverages powerful models for reliable feature extraction and classification. |
| **Data Flow** | **Global In-Memory Model** | The active SVM model is loaded directly into RAM, eliminating the high latency associated with disk I/O during prediction. |
| **Update Mechanism** | **Hot Reload Endpoint** | Allows instantaneous model updates (from AWS S3) in memory without service interruption. |

---

## 2. 🗺️ Project Structure and File Breakdown

The code organization reflects the division of labor:

project-root/ ├── app.py # ➡️ ENTRY POINT & Global Configuration (Registers all controllers) ├── services.py # ➡️ SERVICE LAYER (AI Logic, Data Management, S3 I/O) └── controllers/ # ➡️ CONTROLLER LAYER (HTTP Interface) ├── training_controller.py └── prediction_controller.py


### File Responsibilities Detailed

| File | Type | Key Responsibilities |
| :--- | :--- | :--- |
| **`app.py`** | **Entry Point** | **Initial Setup:** Initializes Flask, registers all controllers, and triggers the initial model load from S3. Hosts the crucial **Hot Reload Endpoint**. |
| **`services.py`** | **Service/Model** | **The Brain:** Handles complex tasks: **AWS S3 I/O** (`boto3`), **AI Processing** (`FaceNet`, **MTCNN**), and **Model State Management** (loading/updating the global in-memory SVM model). |
| **`controllers/`** | **Controller** | **The Gatekeeper:** Validates incoming HTTP requests, calls the necessary function in `services.py`, and formats the output into a JSON response. |

---

## 3. 🧠 Core Logic Flow and Service Integration

### A. Feature Extraction Pipeline Steps

1.  **Preprocessing:** Uses **MTCNN** and advanced **OpenCV** filters to normalize the face.
2.  **Embedding Generation:** The prepared face is passed to the **FaceNet** model, generating the unique **512-D embedding**.

### B. Training Service (`/retrain`)

1.  **Input & Accumulation:** Receives a batch of images and merges them with the historical dataset.
2.  **Retraining:** The **SVM** model is retrained on the cumulative dataset.
3.  **Publishing:** The new model is **uploaded directly to AWS S3**.

### C. Prediction Service (`/predict`)

1.  **Model Access:** Uses the **in-memory SVM model** loaded from RAM (`services.py`).
2.  **Inference:** Performs face identification and returns the predicted identity and confidence score.

---

## 4. 🌐 API Endpoints and Usage

The application operates as a single unit on one port (e.g., 5000) with two distinct services:

### 1. Training Endpoint

| Detail | Value |
| :--- | :--- |
| **Route** | `/retrain` |
| **Method** | `POST` |
| **Purpose** | Initiates batch retraining and publishes the updated model to S3. |
| **Input Format** | `form-data` with keys: `images` (Multiple Files), `label` (Text). |

### 2. Prediction Endpoint

| Detail | Value |
| :--- | :--- |
| **Route** | `/predict` |
| **Method** | `POST` |
| **Purpose** | Performs instantaneous identity inference using the active in-memory model. |
| **Input Format** | `form-data` with key: `image` (Single File). |

**Success Response Example:**

```json
{
  "confidence": "0.98",
  "prediction": "Will_Smith"
}
```
5. ☁️ Cloud Integration (AWS)
Hot Reload Trigger: The /your-secret-reload-path-12345 endpoint ensures that the Prediction Server downloads and loads the latest model from S3 immediately after the Training Server has finished its update.

Persistence: The system uses AWS S3 as the central storage mechanism for all model files (svm_model.pkl).

6. 🧑‍💻 Author
Built and maintained by Mohamed Ashraf.
