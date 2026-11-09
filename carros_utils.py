"""Preprocesamiento, arquitecturas e inferencia del proyecto (reutilizable fuera del cuaderno)."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models
from torchvision import transforms as T

TAM = 224                 # resolución de entrada de todos los modelos
LADO_MAX_CACHE = 320      # reducción inicial (conserva proporción); ver justificación en la Fase 1

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
VIT_MEAN, VIT_STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
VIT_CHECKPOINT = "google/vit-base-patch16-224"

# Normalización que espera cada modelo preentrenado
NORMALIZACION = {
    "resnet_mlp": (IMAGENET_MEAN, IMAGENET_STD),   # torchvision ResNet50_Weights.IMAGENET1K_V2
    "resnet_ft":  (IMAGENET_MEAN, IMAGENET_STD),
    "vit_ft":     (VIT_MEAN, VIT_STD),             # procesador de google/vit-base-patch16-224
}

# ---------------------------------------------------------------- metadatos de clase
MARCAS_COMPUESTAS = ["AM General", "Aston Martin", "Land Rover"]

def extraer_marca(nombre):
    for m in MARCAS_COMPUESTAS:
        if nombre.startswith(m):
            return m
    return nombre.split()[0]

def extraer_carroceria(nombre):
    n = nombre.lower()
    if "convertible" in n: return "Convertible"
    if "minivan" in n:     return "Minivan"
    if "van" in n.split(): return "Van"
    if " cab" in n or " sut" in n: return "Pickup"
    for t in ["suv", "sedan", "coupe", "hatchback", "wagon"]:
        if t in n.split():
            return "SUV" if t == "suv" else t.capitalize()
    return "Otro"

# ---------------------------------------------------------------- preprocesamiento
def reducir(img, lado_max=LADO_MAX_CACHE):
    """Corrige la orientación EXIF, convierte a RGB y reduce la imagen para que su lado mayor
    mida como máximo `lado_max` píxeles, conservando la proporción. No amplía imágenes pequeñas."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = lado_max / max(w, h)
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    return img

class Letterbox:
    """Escala la imagen para que quepa completa en un cuadrado TAM x TAM y rellena el resto
    con un color neutro. Conserva la proporción y no recorta el vehículo."""
    def __init__(self, tam=TAM, relleno=(124, 116, 104)):
        self.tam, self.relleno = tam, relleno
    def __call__(self, img):
        w, h = img.size
        s = self.tam / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        lienzo = Image.new("RGB", (self.tam, self.tam), self.relleno)
        lienzo.paste(img, ((self.tam - img.size[0]) // 2, (self.tam - img.size[1]) // 2))
        return lienzo

def geometria(modo, tam=TAM):
    if modo == "deformar":   # fuerza 224x224 (deforma la proporción)
        return T.Resize((tam, tam), interpolation=T.InterpolationMode.BICUBIC)
    if modo == "recorte":    # lado menor a 224 + recorte central (puede cortar el frente/la cola del carro)
        return T.Compose([T.Resize(tam, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(tam)])
    if modo == "letterbox":  # imagen completa + relleno
        return Letterbox(tam)
    raise ValueError(modo)

def construir_transform(arq, modo_geom="deformar", entrenamiento=False):
    """Transformación de imagen PIL (ya reducida) -> tensor normalizado 3x224x224."""
    mean, std = NORMALIZACION[arq]
    pasos = [geometria(modo_geom)]
    if entrenamiento:   # aumentos SOLO para entrenamiento, aplicados al vuelo (no se guardan)
        pasos += [T.RandomHorizontalFlip(p=0.5),
                  T.RandomRotation(degrees=10),
                  T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
    pasos += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(pasos)

# ---------------------------------------------------------------- arquitecturas
class MLPCabeza(nn.Module):
    """Red densa sobre un vector de tamaño fijo (embedding)."""
    def __init__(self, entrada, n_clases, oculta=512, dropout=0.3):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(entrada, oculta), nn.BatchNorm1d(oculta), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(oculta, n_clases))
    def forward(self, x):
        return self.red(x)

def resnet50_base(pretrained=True):
    pesos = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    return models.resnet50(weights=pesos)

class ResNetMLP(nn.Module):
    """Arquitectura A: ResNet-50 congelada como extractor de embeddings (2048) + MLP entrenable."""
    def __init__(self, n_clases, oculta=512, dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone = resnet50_base(pretrained)
        self.backbone.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.cabeza = MLPCabeza(2048, n_clases, oculta, dropout)
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()      # el backbone congelado siempre en modo evaluación (BatchNorm fijo)
        return self
    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
        return self.cabeza(f)

def crear_resnet_ft(n_clases, dropout=0.3, pretrained=True):
    """Arquitectura B: ResNet-50 completa con ajuste fino (fine-tuning)."""
    m = resnet50_base(pretrained)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, n_clases))
    return m

class ViTClasificador(nn.Module):
    """Arquitectura C: Vision Transformer ViT-B/16 con ajuste fino."""
    def __init__(self, n_clases, pretrained=True):
        super().__init__()
        from transformers import ViTConfig, ViTForImageClassification
        if pretrained:
            self.m = ViTForImageClassification.from_pretrained(
                VIT_CHECKPOINT, num_labels=n_clases, ignore_mismatched_sizes=True)
        else:   # solo la estructura; los pesos se cargan después con load_state_dict
            cfg = ViTConfig.from_pretrained(VIT_CHECKPOINT)
            cfg.num_labels = n_clases
            cfg.id2label = {i: str(i) for i in range(n_clases)}
            cfg.label2id = {str(i): i for i in range(n_clases)}
            self.m = ViTForImageClassification(cfg)
    def forward(self, x):
        return self.m(pixel_values=x).logits

def crear_modelo(arq, n_clases, dropout=0.3, pretrained=True):
    if arq == "resnet_mlp": return ResNetMLP(n_clases, dropout=dropout, pretrained=pretrained)
    if arq == "resnet_ft":  return crear_resnet_ft(n_clases, dropout=dropout, pretrained=pretrained)
    if arq == "vit_ft":     return ViTClasificador(n_clases, pretrained=pretrained)
    raise ValueError(arq)

# ---------------------------------------------------------------- inferencia
@torch.no_grad()
def predecir(modelo, img, transform, nombres, k=3, device="cpu"):
    """Aplica el MISMO pipeline de evaluación del cuaderno a una imagen PIL original.
    Devuelve el top-k [(clase, probabilidad)] y el vector completo de probabilidades."""
    modelo.eval()
    x = transform(reducir(img)).unsqueeze(0).to(device)
    p = torch.softmax(modelo(x).float(), dim=1)[0].cpu().numpy()
    top = np.argsort(p)[::-1][:k]
    return [(nombres[i], float(p[i])) for i in top], p