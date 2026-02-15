# Deployment Guide — Render.com

This guide explains how to deploy the RL vs LLM Trading System to **Render.com** (free tier available).

## Option 1: Render.com (Recommended)

1.  **Push your code to GitHub**
    Make sure your repository is public or private on GitHub.

2.  **Create a Web Service on Render**
    - Sign up / Log in to [Render.com](https://render.com).
    - Click **New +** -> **Web Service**.
    - Connect your GitHub repository.

3.  **Configure the Service**
    - **Name**: `rl-llm-trader` (or similar)
    - **Region**: US East (N. Virginia) is best for stock market latency.
    - **Branch**: `main`
    - **Runtime**: **Docker** (Render will auto-detect the `Dockerfile`).
    - **Instance Type**: **Free** (512MB RAM) is sufficient for initial testing. Upgrade to **Starter** ($7/mo) if you see "Out of Memory" crashes.

4.  **Environment Variables**
    Click **Advanced** or **Environment** and add the following keys from your local `.env`:
    - `GROQ_API_KEY`: `gsk_...`
    - `GEMINI_API_KEY`: (If you still use Gemini for anything)

5.  **Deploy**
    - Click **Create Web Service**.
    - Render will build the Docker image (takes ~5-10 mins).
    - Once "Live", click the URL (e.g., `https://rl-llm-trader.onrender.com`) to see your dashboard!

> **Note**: The **Trading Scheduler** runs in the background of the same container. It will trigger trades at 10:00, 13:00, and 15:00 EST automatically as long as the service is running. On the Free tier, Render spins down inactive web services after 15 mins of inactivity. **For reliable 24/7 trading, you MUST use a paid instance ($7/mo) to keep it awake.**

---

## Option 2: Local Docker (Testing)

Test the container on your machine before deploying:

1.  **Build the image**:
    ```bash
    docker build -t traders .
    ```

2.  **Run the container**:
    ```bash
    docker run -p 8501:8501 --env-file .env traders
    ```

3.  **Access**:
    Open `http://localhost:8501` to view the dashboard.
