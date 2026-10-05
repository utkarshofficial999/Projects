# 🤖 GitAgentic: Autonomous Daily AI Engineer & GitHub Committer

> An autonomous agent that builds, tests, documents, and regularly commits an entire Agentic AI project to your GitHub profile, executing discrete engineering milestones at randomized times every 24 hours.

---

## ✨ Features

- 🧠 **Autonomous Engineering Lifecycle**: Given a high-level topic (e.g. *"Multi-Agent Workflow Engine with Self-Reflection"*), the agent architecturally plans a 15–25 step sequential engineering roadmap.
- ⏰ **Randomized 24-Hour Schedule**: Runs every ~24 hours with configurable random jitter (e.g., between 18h and 28h) so your daily GitHub commit streak appears natural at varying times of day.
- ⚡ **Multi-LLM Compatible**: Built on an OpenAI-compatible interface:
  - **Groq** (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) — ultra-fast & free tier friendly!
  - **Ollama** (`qwen2.5-coder`, `llama3.2`) — 100% free, local, private, no API key needed.
  - **OpenAI** (`gpt-4o`, `gpt-4o-mini`).
  - **DeepSeek** (`deepseek-chat`).
- 🛠️ **Production-Ready Code**: Generates full working code, unit tests (`pytest`), docstrings, and clean architecture without hollow `# TODO` placeholders.
- 📝 **Conventional Git Commits**: Stages code and generates conventional commit messages (e.g. `feat(engine): add dynamic tool registry and schema generator`).
- ☁️ **Dual Deployment Options**:
  - **Local Daemon**: Runs on your machine with live countdown and status tables.
  - **GitHub Actions**: Runs 24/7 in the cloud without needing your PC turned on.

---

## 🚀 Quick Start (Local Setup)

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd "github aagetn"
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy the example environment template:
```bash
cp .env.example .env
```
Edit `.env` with your API key and desired project topic:
```env
# Example for Groq
LLM_BASE_URL="https://api.groq.com/openai/v1"
LLM_API_KEY="gsk_your_groq_api_key_here"
LLM_MODEL="llama-3.3-70b-versatile"

# Your Agentic AI project topic
PROJECT_TOPIC="Autonomous Multi-Agent Workflow Engine with Tool Calling, Dynamic Memory and Self-Reflection"

# Target workspace to commit to
TARGET_PROJECT_PATH="./target_project"
GIT_AUTO_PUSH=true
```

*(If using **Ollama**, simply set `LLM_BASE_URL="http://localhost:11434/v1"` and `LLM_MODEL="qwen2.5-coder:7b"` with `LLM_API_KEY="ollama"`!)*

---

## 💻 CLI Commands

### 1. Preview the Roadmap
See the agent's architectural plan before executing any code:
```bash
python main.py --init-roadmap
```

### 2. Run a Single Milestone (Immediate Run)
Runs the next pending milestone, creates files, writes tests, commits, and pushes:
```bash
python main.py --run-once
```

### 3. Check Current Status & Progress
View completion percentage, table of completed and pending milestones, and last commit:
```bash
python main.py --status
```

### 4. Start the 24-Hour Randomized Daemon
Starts the continuous loop on your machine. It executes a step, then randomizes the wait time between 18 and 28 hours before the next commit:
```bash
python main.py --start-daemon
```

---

## ☁️ Deploying on GitHub Actions (Run 24/7 Without Keeping PC On)

The included `.github/workflows/daily_agent.yml` lets GitHub automatically run the agent once a day.

### Setup Instructions:
1. Push this repository to your GitHub account:
   ```bash
   git add .
   git commit -m "feat: initialize GitAgentic system"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. In your GitHub repository:
   - Go to **Settings** > **Secrets and variables** > **Actions**.
   - Add the following Repository Secrets:
     - `LLM_API_KEY`: Your Groq / OpenAI API key.
     - *(Optional)* `LLM_BASE_URL`: (Defaults to `https://api.groq.com/openai/v1`).
     - *(Optional)* `LLM_MODEL`: (Defaults to `llama-3.3-70b-versatile`).
     - *(Optional)* `PROJECT_TOPIC`: Custom topic if different from default.
3. Under **Settings** > **Actions** > **General** > **Workflow permissions**:
   - Select **Read and write permissions** (so the action can push commits).
4. GitHub Actions will now trigger daily on schedule with randomized jitter, committing new features under your GitHub account!

---

## 📂 Project Architecture

```
├── .github/workflows/
│   └── daily_agent.yml       # Cloud 24-hr scheduled GitHub Action
├── src/
│   ├── config.py             # Settings, environment, and validation
│   ├── llm_client.py         # Universal OpenAI-compatible LLM client
│   ├── roadmap_engine.py     # Generates 15-25 step architecture & tracks progress
│   ├── coder_agent.py        # Generates production code, tests, and commit messages
│   ├── git_committer.py      # Git staging, status, commit, and remote push
│   └── scheduler.py          # Randomized 24-hour daemon runner
├── target_project/           # Dedicated directory where the AI builds the project
├── main.py                   # Central CLI interface
├── requirements.txt          # Python dependencies
└── .env.example              # Configuration template
```

---

## 🛡️ License
MIT License. Built for autonomous AI exploration and open-source contributions.
