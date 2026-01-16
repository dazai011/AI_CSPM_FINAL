# AI_CSPM

# AI-Powered Cloud Security Posture Management (CSPM)

## Project Overview
AI_CSPM_FINAL is an AI-powered Cloud Security Posture Management (CSPM) system that analyzes cloud environments (AWS, Azure, and GCP) to detect security misconfigurations, risks, and compliance issues. The project integrates rule-based security checks with a local Large Language Model (LLM) to provide human-readable risk explanations and remediation guidance.

This project is developed as a Final Year Academic Project and follows real-world DevSecOps and cloud security best practices.

---

## Project Objectives
- Detect cloud security misconfigurations
- Perform automated cloud posture analysis
- Use AI to explain risks and remediation steps
- Support multi-cloud environments
- Demonstrate secure and scalable cloud governance concepts

---

## 🏗️ Project Structure

AI_CSPM_FINAL/
│
├── ai/
│ ├── ai_engine.py
│ ├── llm_runner.py
│ ├── rule_engine.py
│ ├── analyze_utils.py
│ └── model/ # LLM model directory (model not included)
│
├── cloud_providers/
│ ├── aws.py
│ ├── azure.py
│ └── gcp.py
│
├── scanners/
│ ├── aws_scanner.py
│ ├── azure_scanner.py
│ └── gcp_scanner.py
│
├── app.py # Main CLI entry point
├── requirements.txt
├── .gitignore
└── README.md


---

## ☁️ Supported Cloud Providers
- AWS
- Microsoft Azure
- Google Cloud Platform (GCP)

Note : Integrated only aws for now !
---

##  AI & LLM Integration

This project uses a local Large Language Model (LLM) to generate security risk explanations and remediation suggestions.

### ⚠️ Important Note
Due to GitHub file size limitations, LLM model files are NOT included in this repository.

### 📥 LLM Model Setup
1. Download a GGUF-compatible LLM model (example: `mistral-7b-instruct-v0.3-q4_k_m.gguf`)
2. Place the model inside the following directory:


## 🛠️ Installation & Setup (Step by Step)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/dazai011/AI_CSPM_FINAL.git
cd AI_CSPM_FINAL


Create and Activate Virtual Environment

python -m venv venv


Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add LLM Model

Place the downloaded .gguf model file into:

ai/model/

▶️ Running the Application

Start the CSPM interactive shell:

python app.py

🔐 Cloud Authentication

When running the application, you will be prompted to enter credentials for the selected cloud provider.

AWS

Access Key ID

Secret Access Key

Region

Azure

Tenant ID

Client ID

Client Secret

Subscription ID

GCP

Path to Service Account JSON file

Credentials are never stored and are used only during runtime.

 Key Features

Interactive CSPM command-line interface

Multi-cloud security scanning

Rule-based security evaluation

AI-powered risk analysis and explanation

Secure credential handling

Modular and extensible architecture

 Excluded from Repository

The following are intentionally excluded:

Virtual environment (venv/)

LLM model files (.gguf)

Cache files (__pycache__/)

🎓 Academic Note

This project is submitted as a Final Year Undergraduate Project and demonstrates practical implementation of cloud security, AI-assisted security analysis
