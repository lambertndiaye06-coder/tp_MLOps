FROM python:3.12.5

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Créer le fichier vide pour que Docker sache que c'est un fichier et non un dossier
RUN touch /app/src/mlops_tp/crx.data

WORKDIR /app/src/mlops_tp

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]