
FROM python:3.8-slim

# Add non-root user
RUN addgroup --system appgroup && adduser --system --group appuser

WORKDIR /app

# Copy only the requirements to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining files
COPY . .

# Change user from root
USER appuser

# Expose port and set environment variables if needed
EXPOSE 8000
ENV UVICORN_WORKERS=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
