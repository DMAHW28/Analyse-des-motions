import torch
from utils import test_models
from peft import LoraConfig, get_peft_model
from models import  TextClassifierTransformer
from fastapi import FastAPI, Request, HTTPException
from transformers import BertTokenizer, BertForSequenceClassification

# --------- Config ----------
DEVICE = torch.device('cpu')
MODEL_DIR = "../models"
EMO_DICO = {
    0: "triste",
    1: "joie",
    2: "amour",
    3: "colère",
    4: "peur",
    5: "surprise",
}

# --------- Load Models ----------
def load_models(device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = len(tokenizer.get_vocab())
    output_dim = len(EMO_DICO)

    # TRANSFORMER
    transformer = TextClassifierTransformer(
        vocab_size=vocab_size,
        output_dim=output_dim,
        num_layers=1,
        n_head=1,
        d_model=64,
        dim_feedforward=128,
        dropout=0.5,
        max_len=vocab_size
    )
    transformer.load_state_dict(torch.load(f"{MODEL_DIR}/transformers_model_fn.pth", map_location=device))
    transformer.eval().to(device)

    # BERT
    bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=output_dim)
    bert.load_state_dict(torch.load(f"{MODEL_DIR}/bert_model.pth", map_location=device))
    bert.eval().to(device)

    # BERT LORA
    bert_lora = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=output_dim)
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="SEQ_CLS")
    bert_lora = get_peft_model(bert_lora, lora_config)
    bert_lora.load_state_dict(torch.load(f"{MODEL_DIR}/lora_bert_model.pth", map_location=device))
    bert_lora.eval().to(device)

    # Centraliser les modèles
    models = {
        "Bert": (bert, True),
        "Bert Lora": (bert_lora, True),
        "Transformer": (transformer, False)
    }

    return models, tokenizer

MODELS, TOKENIZER = load_models(DEVICE)

# --------- FastAPI ----------
app = FastAPI(title="Emotion Detection API", version="1.0")

@app.post("/predict")
async def root(request: Request):
    data = await request.json()
    text = data["text"]
    method = data["method"]

    if method not in MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid method. Choose one of: {list(MODELS.keys())}")

    model, bert_like = MODELS[method]
    with torch.inference_mode():
        prediction = test_models(text, model, TOKENIZER, DEVICE, bert_like)
    emotion = EMO_DICO[prediction.item()]
    return {"emotion": emotion}