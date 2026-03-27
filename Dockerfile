FROM python:3.11

WORKDIR /app

# 1. Copier requirements.txt EN PREMIER
COPY requirements.txt .

# 2. Installer numpy avant le reste pour éviter les conflits binaires
RUN pip install --no-cache-dir --upgrade pip numpy pandas

# 3. Installer le reste des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le code source
COPY src/ ./src/

RUN touch /app/src/mlops_tp/crx.data

WORKDIR /app/src/mlops_tp

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]