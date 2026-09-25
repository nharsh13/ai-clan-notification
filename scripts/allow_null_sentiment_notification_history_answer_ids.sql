ALTER TABLE public.sentiment_notification_history
    ALTER COLUMN response_id DROP NOT NULL,
    ALTER COLUMN answer_id DROP NOT NULL;
