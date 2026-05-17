FROM python:3.10-slim
WORKDIR /app
RUN pip install streamlit requests
COPY dashboard.py .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0"]