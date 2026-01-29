# Agents with LangGraph

This repository contains code for learning and building agents with **LangGraph**. The content is created for **proof-of-concept and training purposes**, with the goal of getting hands-on experience designing and implementing agent-based systems.

The project focuses on experimentation, learning, and exploring different architectural patterns rather than production-ready solutions.

---

## Inspiration & Learning Resources

The ideas, patterns, and approaches used in this repository are inspired by a variety of online courses, tutorials, and documentation from platforms such as **Coursera**, **Udacity**, and other publicly available technical resources.

This project is **not a direct copy** of any single course or material. Instead, it reflects my own implementations, experiments, and adaptations created through learning, practice, and exploration of agent frameworks and LLM-based systems.

---

## How to Run

Some of the files can be run as **Jupyter Notebooks** directly within **Visual Studio Code** for an interactive experience.  
Other files are intended to be executed from an **Anaconda terminal**.

Make sure the required dependencies are installed before running the examples.

---

## Environment Variables

Some components of this project require API keys or tokens (free or paid), such as external research or tool integrations (for example, the Tavily research tool).

To securely manage sensitive values, this project uses a `.env` file.

### Setup

1. Create a `.env` file in the root of the repository:
   ```bash
   touch .env

2. Add your API keys or tokens:
TAVILY_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here

3. Ensure the `.env` file is not committed to version control by adding it to `.gitignore`:
.env

---

### LLM Usage

For testing and local experimentation, this project uses small, free Large Language Models (LLMs) hosted locally via LM Studio

For production environments or more advanced testing scenarios, it is recommended to use a more scalable and robust serving solution such as vLLM