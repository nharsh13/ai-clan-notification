<<<<<<< HEAD
# AI-CLAN Notification

AI-CLAN exposes one common notification pipeline for three business flows:

- `performance`: weakest KII to a language-matched semantic video recommendation
- `engagement`: CLAN response-rate notification using the 60% rule
- `sentiment`: one Q&A-based workplace performance notification

## Local setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

4. Add the real `OPENAI_API_KEY`, `DATABASE_URL`, and notification provider URL
to `.env`. Never commit `.env`.

5. Start the API from this directory:

```powershell
uvicorn app.main:app --reload
```

Startup validates the database URL and LLM key. OpenAI is the default
embedding provider when `OPENAI_API_KEY` is present. To use local Sentence
Transformers embeddings, set `EMBEDDING_PROVIDER=sentence_transformers`.

## Run a notification

6. Trigger the performance flow after the API starts:

```powershell
Invoke-RestMethod -Method Post `
	-Uri http://127.0.0.1:8000/notification/send `
	-ContentType "application/json" `
	-Body '{"user_id":953,"flow":"performance","should_send":true}'
```

7. Verify the response has `remote_send_status` set to `sent` and contains a
`video_id`, matching `reference_id`, and a deep link such as `/videos/123`.
The configured notification provider receives that exact video ID as
`reference_id`.

8. When the user clicks the notification, the client uses `deep_link` to open
that exact video route. The video screen resolves the supplied ID and starts
that video instead of opening a generic video list.

## Test

Run the complete mocked test suite with:

```powershell
pytest
```

Tests mock external database, embedding, OpenAI, vector-search, and sender
calls. A real end-to-end send additionally requires reachable production
services.

Read a user's performance result:

```http
GET /notification/performance
```

```json
{"user_id": 953}
```

The read-only flow endpoints are:

- `GET /notification/performance`
- `GET /notification/engagement`
- `GET /notification/sentiment`

Each accepts the same JSON request body. Only `POST /notification/send`
generates and sends a personalized notification.

The optional `flow` value is `performance`, `engagement`, or `sentiment`. All
flows produce the same notification object and use the same remote sender.
=======
# ai-clan-notification
>>>>>>> 58b73495156fed04d57f13aebd584ab7ac400484
