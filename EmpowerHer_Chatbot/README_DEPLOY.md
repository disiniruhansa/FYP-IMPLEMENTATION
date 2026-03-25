# Deployment

This project is prepared for free deployment on Hugging Face Spaces using Docker.

## What is included

- `Dockerfile`: builds the React frontend and serves the Flask app with Gunicorn
- `requirements.txt`: Python runtime dependencies
- `.dockerignore`: excludes local virtualenvs, build output, and caches

## Deploy steps

1. Push the `EmpowerHer_Chatbot` folder contents to your GitHub repository.
2. Go to Hugging Face Spaces.
3. Create a new Space.
4. Choose `Docker` as the SDK.
5. Link the Space to your GitHub repository or upload these files.
6. Wait for the image build to finish.
7. Open the Space URL and test the chatbot.

## Runtime notes

- The app listens on port `7860`.
- The frontend is built from `FRONTEND/` during the Docker build.
- Flask serves the built frontend and the `/chat` API from one container.
- The first startup may take longer because transformer models need to download.
