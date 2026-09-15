# AI-CLAN Notification

AI-CLAN exposes one common notification pipeline for three business flows:

- `performance`: weakest KII to a language-matched semantic video recommendation
- `engagement`: CLAN response-rate notification using the 60% rule
- `sentiment`: one Q&A-based workplace performance notification

Start the API from this directory:

```powershell
uvicorn app.main:app --reload
```

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