# Open Source LLM Backend

- Author: Edward He
- Project Start Date: MAY/10/2026

## Summary

This is a backend API Template for Open Source LLMs. If development goes as planned, you can deploy scripts in this repo on any cloud.

## Structure

* main.py: This handles internet requests
* ai.py: This handles model download and inference

**API Routes**

- **Base URL**: Use the host and port from `config.yml` (default `http://0.0.0.0:5000` for cloud use replace with your public domain).

- **GET /**: API info and config summary. Useful for the frontend to display model and endpoint metadata.

- **GET /status**: Returns server status, model id, and active session count. Good for a dashboard health card.

- **GET /health**: Lightweight health check for orchestration and probes.

- **POST /chat**: Main chat endpoint the frontend should call.
	- Request JSON:
		- `message` (string) — user message (required)
		- `session_id` (string) — optional, server will generate if omitted
		- `max_tokens` (int) — optional override for tokens
	- Response JSON:
		- `response` (string) — assistant reply
		- `session_id` (string) — session identifier to persist on the frontend
		- `message_count` (int) — number of messages in session
		- `timestamp` (string) — when response was generated (optional)

- **POST /clear**: Clears a conversation.
	- Request JSON: `{ "session_id": "<id>" }` (optional — generates/uses header if omitted)
	- Response JSON: `{ "status": "cleared", "session_id": "<id>" }`

- **GET /history/<session_id>**: Returns full conversation history for the given session id.

**Example curl requests**

- Chat (send message):

```
curl -X POST http://127.0.0.1:5000/chat \
	-H "Content-Type: application/json" \
	-d '{"message": "Hello, who are you?"}'
```

- Clear conversation:

```
curl -X POST http://127.0.0.1:5000/clear \
	-H "Content-Type: application/json" \
	-d '{"session_id": "your-session-id"}'
```

- Get history:

```
curl http://127.0.0.1:5000/history/your-session-id
```

**Frontend integration notes**

- Store the returned `session_id` on the client (localStorage) and include it for subsequent calls so the server preserves conversation state.
- Respect CORS settings in `config.yml` — update `server.cors_origins` for your frontend domain when deploying to production.
- For production, set `server.host`/`server.port` appropriately and secure the endpoint with a proxy/HTTPS.

See [main.py](main.py) and [ai.py](ai.py) for implementation details and request/response structure.

**Project Overview**

This repository provides a small Flask-based backend to serve an open-source LLM as a chat API. It loads model and server settings from `config.yml`, exposes simple REST endpoints for a chat frontend, and keeps conversation state in-memory per `session_id`.

**Configuration**

- Configuration file: `config.yml` (controls model, inference, server, CORS, and logging)
- Default host: `0.0.0.0` (set in `server.host`)
- Default port: `5000` (set in `server.port`)
- CORS: Controlled by `server.cors_enabled` and `server.cors_origins` — update to your frontend domain in production

When deployed to cloud, set `server.host` and `server.port` appropriately (or run behind a reverse proxy). If you expose the API publicly, place it behind HTTPS and an authentication/proxy layer.

**API URL Use Cases (examples)**

- Local development (default):
	- Base URL: `http://127.0.0.1:5000`
	- Chat: `POST http://127.0.0.1:5000/chat`

- Example cloud deployment (replace `api.example.com` with your domain):
	- Base URL: `https://api.example.com`
	- Chat: `POST https://api.example.com/chat`
	- Health (kubernetes/ALB probe): `GET https://api.example.com/health`

**Frontend integration (recommended)**

- Persist the `session_id` returned from `/chat` in `localStorage` (or cookies) and include it with subsequent requests to preserve context.
- Ensure your frontend origin is allowed by `server.cors_origins` in `config.yml`.

Example `fetch` usage (JavaScript):

```
// sendMessage(message, sessionId)
async function sendMessage(message, sessionId = null) {
	const res = await fetch('https://api.example.com/chat', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ message, session_id: sessionId })
	});

	const data = await res.json();
	// store session_id for later
	localStorage.setItem('chat_session_id', data.session_id);
	return data;
}
```

Quick deployment notes

- Use a production WSGI server (gunicorn, waitress) or containerize with Docker. Run behind an HTTPS reverse proxy (nginx, Traefik).
- For performance and cost, consider using smaller models or running on GPUs; configure `config.yml` `model.device_map` and `dtype` accordingly.

If you'd like, I can add a simple `docker-compose.yml` and a `requirements.txt` entry for production readiness.