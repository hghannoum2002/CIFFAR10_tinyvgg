# IA Embarqué - Classification CIFFAR-10 sur Stm32 (avec volet sécurité)


> **Projet réalisé par :** Rawane AL ZOHBI & Halima GHANNOUM  
> **Formation :** Systéme Embarqué – EMSE-ISMIN 2025  
> **Encadrants :** M.Olivier POTIN, M.Pierre Alain MOEILLIC, M.Kévin HECTOR

# Partie 1 – Amélioration et Exécution du Modèle sur STM32 via Python

## Objectif 
L'objectif est de concevoir une **solution d’IA embarquée de bout en bout** pour la classification **CIFAR-10** (10 classes d’images 32×32), depuis l’entraînement du modèle jusqu’à son **déploiement sur STM32L4R9AI**, en assurant :
- une **communication UART** fiable **PC <-> STM32**,  
- une **évaluation sur cible** (accuracy réelle),  
- un **volet sécurité** : *adversarial examples* (PGD) et **Bit-Flip Attack** (BFA).
Ces étapes visent à valider la compatibilité du modèle avec les contraintes embarquées et à mesurer la précision réelle obtenue sur le matériel.

### Architecture du dépôt
```
IA_Embarqué_Rawane_Halima
├── main/   
| ├─ Brouillon/        #Brouillon pour notre travail, notes....
| ├─ assets/           #Images et résultats  
| └── README.md         # Rapport & documentation complète (ce fichier)    
├── feature/model/    #le model .h5 génèrer et les données xtest et ytest
| └── model             
└── dev/
 ├─Code python/      #les fishiers pythons utilisés pour la communication, génération du modéle
 └── Stm Project/     #notre projet stm32
```

## Méthodologie

Cette section présente, étape par étape, notre démarche technique et réflexive pour mener à bien le projet.  
Elle reprend les points du cahier des charges demandé par le professeur, mais réécrits de manière fluide et professionnelle.

---

### 1. Analyse du modèle existant

Le modèle de départ, fourni par l’encadrant, implémente une architecture **VGG11** adaptée à **CIFAR-10**.  
Le script charge et normalise les images, puis construit un modèle **séquentiel** basé sur plusieurs blocs de convolutions (filtres 3×3, ReLU, Batch Normalization) suivis de **MaxPooling** pour réduire la taille spatiale.

En sortie, deux **couches pleinement connectées** (1024 et 512 neurones) assurent la classification avant la couche **Softmax(10)**.  
Le modèle est entraîné avec **Adam**, puis sauvegardé au format `.h5` pour un déploiement potentiel sur carte STM32.

**Observation principale :**  
Ce réseau, bien que performant, se révèle **trop volumineux** pour une exécution sur microcontrôleur : les couches denses, combinées à `Flatten()`, entraînent un nombre de paramètres très élevé, donc une taille mémoire excessive (≈5 Mo).

---

### 2. Étude du microcontrôleur cible

Nous avons choisi le **STM32L4R9AI**, un microcontrôleur hautes performances doté de :

- **2 Mo de mémoire Flash interne** et **640 Ko de RAM**,  
- une **PSRAM externe de 16 Mbit** et une **Octo-SPI Flash de 512 Mbit**,  
- un **cœur Cortex-M4F** cadencé à **120 MHz** avec **unité FPU**,  
- plusieurs **interfaces de communication** (UART, SPI, I²C, USB, écran intégré),  
- et la compatibilité native avec **STM32Cube.AI**, facilitant l’import et la conversion de modèles de deep learning.

Ces caractéristiques en font une plateforme idéale pour **tester l’embarquement d’un CNN compact**, tout en assurant des échanges UART et un affichage rapide des résultats.

---

### 3. Évaluation de l’embarquipossibilité du modèle initial

Nous avons d’abord tenté d’évaluer la faisabilité du déploiement du modèle original sur notre STM32L4R9AI.  
Les résultats de **STM32Cube.AI** ont révélé :

- une **consommation RAM** correcte (< 200 Ko),  
- mais une **taille Flash de plus de 5 Mo**, bien au-delà des 2 Mo disponibles.  

Ainsi, même si la RAM restait suffisante, le stockage interne limitait la possibilité d’embarquer ce modèle.

> **Conclusion :** le modèle initial n’est **pas déployable tel quel**.  
> Nous avons donc choisi de **concevoir un nouveau modèle** plus léger, adapté à la plateforme STM32, tout en maintenant une précision satisfaisante.

---

### 4. Conception de notre nouveau modèle CNN

Notre objectif était de **réduire drastiquement la taille** du réseau sans compromettre la précision.  
Pour cela, plusieurs **optimisations architecturales** ont été introduites :

1. **Remplacement du `Flatten()` par un `GlobalAveragePooling2D()`** :  
   - `Flatten()` transforme les cartes de caractéristiques en un long vecteur, explosant le nombre de paramètres de la couche suivante.  
   - `GlobalAveragePooling2D()` résume chaque carte par sa moyenne → moins de paramètres, moins de mémoire.

2. **Suppression des deux couches denses lourdes (1024 et 512)** :  
   - remplacées par **Dense(64)** puis **Dense(10)**, réduisant le nombre de poids de plusieurs millions à quelques milliers.

3. **Ajout d’un bloc convolutionnel intermédiaire (96 filtres)** :  
   - facilite la transition entre les blocs 64 et 128 filtres, rendant l’extraction de caractéristiques plus fluide et plus généralisable.

4. **Insertion de BatchNormalization et SpatialDropout2D** :  
   - améliore la stabilité et la régularisation sans perte significative de précision.

5. **Ajout du callback `ReduceLROnPlateau`** :  
   - ajuste automatiquement le learning rate, permettant une convergence plus stable.

**Résultats obtenus :**
| Indicateur | Valeur |
|-------------|--------|
| Accuracy test | **≈ 87 %** |
| Taille du modèle | **≈ 1.15 Mo** |
| Complexité (MACs) | **≈ 39 M** |
| RAM utilisée (STM32Cube.AI) | **≈ 148 Ko** |

Ces optimisations permettent désormais un **déploiement complet sur STM32L4R9AI sans quantification**, tout en conservant d’excellentes performances.

---

### 5. Conversion du modèle pour la cible embarquée

La conversion du modèle s’effectue via **STM32Cube.AI**, qui génère automatiquement :
- les fichiers C (`app_x-cube-ai.c`, `cifar10_data.c`),
- les buffers d’entrée/sortie,
- et la configuration des activations.

Aucune étape de quantification n’a été nécessaire :  
la taille finale du modèle (≈1.15 Mo) est compatible avec la **Flash interne**.  
La quantification INT8 reste une option future pour **accélérer les transferts UART** et **réduire encore la Flash**.

---

### 6. Sélection du microcontrôleur

Nous avons conservé le **STM32L4R9AI**, parfaitement adapté à notre modèle compact (1.15 Mo, 148 Ko RAM).  
Sa **Flash de 2 Mo**, son **cœur Cortex-M4F à 120 MHz** et sa **compatibilité avec STM32Cube.AI** offrent un excellent compromis entre performance, simplicité et consommation.  
Il permet d’exécuter le modèle sans quantification ni perte de précision.  

De manière plus générale, le choix d’un microcontrôleur dépend toujours de la **taille mémoire disponible**, de la **puissance de calcul**, de la **compatibilité avec les outils IA**, de la **consommation énergétique** et des **interfaces** nécessaires.  

---

### 7. Intégration et communication UART

Une fois le modèle généré, nous avons intégré le code dans **STM32CubeIDE**.  
La communication PC <-> STM32 s’effectue via **UART (115200 bps)** avec un **handshake simple** :
- le PC envoie `0xAB` ;  
- la carte répond `0xCD 0x00` pour synchronisation.

Chaque image CIFAR-10 (32×32×3 float32) est envoyée via **12 288 octets**.  
Le STM32 effectue l’inférence, puis renvoie **10 probabilités (uint8)** que le PC rééchelonne en [0,1].

Le script Python `Communication.py` :
- gère l’envoi des entrées et la réception des sorties,
- calcule l’accuracy en temps réel,
- et affiche les prédictions successives.

---

### 8. Évaluation et performances

- **Cohérence des prédictions** : parfaite entre PC et STM32.  
- **Accuracy finale observée** : entre **86 % et 89 %** selon le sous-échantillon.  
- **Latence UART (115200 bps, FP32)** : environ **1.07 s/image**.  

L’inférence embarquée est donc fluide, fiable, et conforme à la performance du modèle d’origine.

---

## Résultats
![resultat analyse STM](images/resultat.png)
![synchronization_python](assets/synchronization_python.jpg)

Les résultats montrent une excellente cohérence entre les inférences du modèle embarqué et celles du modèle original entraîné sur PC.  
La précision finale obtenue est de 89 %, confirmant :

- la bonne intégration du modèle CNN sur STM32,  
- et la synchronisation UART parfaitement opérationnelle entre la carte et l'environnement Python.



# Partie 2 — Bit-Flip Attack & Attaque Adversariale sur CNN (CIFAR-10)

##  Objectif
Nous allons appliquer sur notre modèle **CNN entraîné sur CIFAR-10** deux types d'attaques pour évaluer la robustesse :  
- un **Bit-Flip Attack (BFA)** *(attaque sur les poids quantifiés)*,  
- et une **attaque adversariale** *(attaque sur les images d'entrée)*.

---

##  Qu'est-ce que le Bit-Flip Attack ?
Le **Bit-Flip Attack (BFA)** est une attaque matérielle/logicielle visant les poids quantifiés d'un réseau de neurones.  
En modifiant un petit nombre de bits sensibles dans la représentation binaire des poids,  
l'attaquant peut fortement dégrader la précision du modèle — parfois avec très peu de flips.

---

##  Notre protocole
- **Modèle cible** : notre CNN personnalisé entraîné sur **CIFAR-10**.  
- **Quantification** : poids quantifiés sur **8 bits**.  
- **Attaque (BFA)** : exécution de l'algorithme BFA sur les poids quantifiés et mesure de la chute d'exactitude sur le jeu de test.


---

##  Paramètres que nous allons faire varier
Nous testerons l’impact de trois variables de configuration du modèle :  

- **`lr`** — learning rate utilisé pendant l’entraînement.  
- **`clipping_value`** — valeur de clipping appliquée aux poids.  
- **`randbet`** — flag binaire pour activer/désactiver la protection RandBET.

Nous exécuterons la BFA sur les différentes combinaisons de ces paramètres afin de comparer la robustesse des modèles.

---

##  Attaques adversariales
L’attaque adversariale consiste à ajouter à l’image d’entrée une **perturbation spécialement conçue** — optimisée pour maximiser la perte du modèle sous une contrainte de norme. Autrement dit, **ce n’est pas du bruit aléatoire** : c’est une modification calculée pour tromper le modèle.

###  Paramètres d’attaque à balayer
- **`step`** — amplitude du pas de mise à jour 
*.  
- **`iterations`** — nombre d’itérations de la méthode .


---

##  Résultats 

 **Fichiers utiles :**
- `train_cnn.py` — entraînement du modèle avec paramètres `lr`, `clipping_value`, `randbet`.  
- `bfa_cnn.py` — script d’exécution du Bit-Flip Attack.  
- `adversarial_example.ipynb` — notebook d’expérimentation des attaques adversariales (paramètres `step` et `iterations`).  

---
## Installation

**Prérequis**
- Python ≥ 3.8  
- TensorFlow, Keras, NumPy, Matplotlib  
- STM32CubeIDE + pack **X-CUBE-AI** installé  
- (Optionnel) Git pour cloner le dépôt

**Étapes**
```bash
# Cloner le dépôt GitLab
$ git clone https://gitlab.emse.fr/rawane.alzohbi/IA_Embarqué_Rawane_Halima.git

# Accéder au dossier du projet
$ cd IA_Embarqué_Rawane_Halima

# (Optionnel) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate     # sous Linux/macOS
venv\Scripts\activate        # sous Windows

```
### Utilisation

- **Exécution côté Python (PC hôte)**  
  Le script `cifar10_model.py` permet d’entraîner et d’évaluer notre modèle CNN sur le jeu de données **CIFAR-10**.  
  Une fois l’entraînement terminé, le modèle est sauvegardé au format **`.h5`** (`CIFAR10_VGG11_simple.h5`) ainsi que les fichiers de test `CIFAR10_xtest.npy` et `CIFAR10_ytest.npy`.

- **Déploiement sur STM32 (partie embarquée)**  
  Le modèle `.h5` est importé dans **STM32Cube.AI**, qui génère automatiquement le code C correspondant (`app_x-cube-ai.c`, `cifar10_data.c`, etc.).  
  Ce code est ensuite intégré et compilé dans **STM32CubeIDE** pour exécuter le modèle directement sur la carte **STM32L4R9AI**.

- **Communication UART (liaison PC <-> STM32)**  
  Le script Python `Communication.py` établit une communication série (UART) avec la carte.  
  Il envoie les images CIFAR-10 normalisées (32×32×3) vers le STM32, récupère les prédictions du modèle embarqué, puis affiche l’**accuracy** obtenue côté PC.  
  Cette liaison permet de comparer en temps réel les performances du modèle embarqué et du modèle original entraîné sur PC.

- **Volet sécurité (analyse de robustesse)**  

---

