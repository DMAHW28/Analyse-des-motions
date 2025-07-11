# Analyse-des-motions

Ce projet propose une API REST développée avec FastAPI ainsi qu'une application web Streamlit pour la détection des émotions dans des textes, en utilisant des modèles de traitement du langage naturel (NLP).

Trois modèles ont été entraînés sur le jeu de données **Emotion** disponible sur Hugging Face : un BERT fine-tuné, un BERT fine-tuné avec LoRA, et un Transformer entraîné from scratch avec PyTorch.

Le projet permet de :

- Analyser un texte et prédire l’émotion associée.
- Choisir le modèle de prédiction via l’API ou l’interface web.
- Exposer les modèles entraînés pour un usage en production.

## Gestion des données

Le projet utilise le jeu de données Emotion de Hugging Face (`dair-ai/emotion`). Ce dataset est automatiquement téléchargé et converti en fichiers CSV par la fonction `create_dataset` située dans le module `src.preprocessing.py`.

Les fichiers générés sont :

- `train_dataset.csv`
- `test_dataset.csv`
- `validation_dataset.csv`

Ces fichiers contiennent les textes ainsi que leurs labels d’émotions associés.

## Installation

Installez les dépendances via pip :

```bash
pip install -r requirements.txt
```

## Lancer le projet

### Lancer l’API
```bash
cd src
uvicorn api:app --reload
```

Utilisation de l’API
Envoyer une requête POST à l’endpoint /predict :
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text": "I am so happy today!", "method": "Bert"}'
```

### Lancer l’application Streamlit
```bash
cd src
streamlit run web_api.py
```

## Modèles entraînés et performances
Trois modèles ont été entraînés sur le jeu de données Hugging Face - Emotion. Les résultats obtenus sur l’ensemble de test sont les suivants :

| Modèle                   | Accuracy | F1-Score |
| ------------------------ |----------| -------- |
| BERT fine-tuné           | 93.0%    | 0.91     |
| BERT LoRA fine-tuné      | 91.0%    | 0.90     |
| Transformer from scratch | 89.0%    | 0.84     |

Le jeu de données contient 6 étiquettes : triste, joie, amour, colère, peur, surprise

## Technologies utilisées
 - Python 3.10
 - PyTorch
 - Hugging Face Transformers
 - PEFT (LoRA)
 - FastAPI
 - Streamlit
 - Uvicorn
 - Requests

## Auteur
Développé par DMAHW