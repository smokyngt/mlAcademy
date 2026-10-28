---
id: dl.04
title: PyTorch en pratique
summary: Tenseurs et dtypes, autograd, nn.Module, DataLoader, la boucle d'entraînement standard, devices, précision mixte et points de sauvegarde, chacun mis en correspondance avec le réseau NumPy que tu as écrit à la main.
minutes: 80
teaches: [dl.pytorch]
requires: [nn.training, nn.autograd, nn.backprop]
sources:
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/11-intro-to-pytorch/docs/en.md (tensors, dtype table, devices, autograd rules, nn.Module, losses and optimisers, the five-line loop, Dataset/DataLoader, GPU and mixed precision, common bugs). READ."
  - "LLMs-from-scratch/appendix-A/01_main-chapter-code/code-part1.ipynb (A.2 tensors and from_numpy memory sharing, A.3-A.4 computation graph and autograd example, A.5 NeuralNetwork module and parameter count, A.6 Dataset/DataLoader, A.7 training loop and compute_accuracy, A.8 save/load state_dict). READ."
  - "LLMs-from-scratch/appendix-A/01_main-chapter-code/code-part2.ipynb (A.9 moving tensors to a device and the device-aware compute_accuracy). READ. DDP-script.py (multi-GPU) was NOT opened."
  - "LLMs-from-scratch/appendix-D/01_main-chapter-code/appendix-D.ipynb (D.1 warmup, D.2 cosine decay, D.3 gradient clipping with clip_grad_norm_, D.4 the extended training function). READ."
---

## Où cela s'insère

À travers [[f2n.10]] et [[f2n.11]], tu as construit à la main un réseau complet : passe avant, passe arrière dérivée sur papier, un optimiseur, une boucle d'entraînement. C'était le but de ces leçons, et c'est pourquoi PyTorch ne te paraîtra pas magique. Mais personne n'entraîne un vrai modèle ainsi. Un framework t'apporte trois choses que tu ne pourrais raisonnablement pas écrire toi-même : la différentiation automatique pour toute composition d'opérations, des noyaux de calcul rapides sur CPU et GPU, et un écosystème de couches, pertes, optimiseurs et outils de données testés. Cette leçon parcourt les pièces dans l'ordre où tu les rencontres, et pour chacune montre la version à partir de zéro qu'elle remplace. L'exercice termine la boucle en prouvant numériquement que PyTorch calcule la même perte et les mêmes gradients que ton réseau NumPy.

## L'idée en mots simples

Pense au réseau à partir de zéro comme à une voiture construite à la main, et à PyTorch comme à la même voiture construite en usine.

- Un **tenseur** est un tableau NumPy qui retient trois faits supplémentaires : quel format de nombre il utilise (dtype), où il vit (CPU ou un GPU particulier), et si quelqu'un veut sa dérivée.
- **Autograd** est un magnétophone. Pendant que la passe avant s'exécute, chaque opération est écrite sur la bande. Appeler `loss.backward()` rejoue la bande à l'envers et applique la règle de la chaîne à chaque étape : exactement la passe arrière que tu as dérivée, mais générée au lieu d'être écrite à la main.
- Un **nn.Module** est une boîte étiquetée de paramètres plus une fonction `forward`. Il enregistre ses paramètres pour que l'optimiseur puisse les trouver sans que tu tiennes une liste.
- Un **DataLoader** est la personne qui mélange le jeu de données, le découpe en mini-batches et te les remet.
- La **boucle d'entraînement** est les mêmes cinq lignes que tu as déjà écrites : effacer les anciens gradients, passe avant, perte, passe arrière, mise à jour.
- **Devices et précision** concernent la vitesse et la mémoire : place les tenseurs là où se trouve le matériel rapide, et utilise moins de bits par nombre quand tu peux te le permettre.
- Un **point de sauvegarde** (checkpoint) est un fichier de sauvegarde, pour que l'entraînement survive à un plantage et que le modèle puisse être utilisé plus tard.

## Correspondance entre la version NumPy et PyTorch

| À partir de zéro ([[f2n.10]], [[f2n.11]], [[dl.01]]) | PyTorch |
|---|---|
| `X @ W + b` avec `W` de forme `(n_in, n_out)` | `nn.Linear(n_in, n_out)`, poids stocké en `(n_out, n_in)` |
| `np.maximum(0, Z)` | `nn.ReLU()` ou `torch.relu` |
| softmax + entropie croisée écrits à la main, passe arrière `(P - Y)/B` | `F.cross_entropy(logits, labels)` sur des logits bruts et des étiquettes entières, moyenne sur le batch |
| ton `forward_backward` | passe avant sous autograd, puis `loss.backward()` |
| liste de tableaux `params` | `model.parameters()`, `model.state_dict()` |
| `sgd_step`, classe `Adam` | `torch.optim.SGD`, `Adam`, `AdamW` |
| mélange et découpage manuels | `Dataset` + `DataLoader(shuffle=True, batch_size=...)` |
| initialisation de He à la main | `nn.init.kaiming_normal_(w, nonlinearity="relu")` |
| test de gradient | `torch.autograd.gradcheck` en float64, ou comparaison avec tes gradients NumPy comme dans l'exercice |
| fonction warmup/cosinus ([[dl.01]]) | fixer `param_group["lr"]` à chaque pas, ou un `torch.optim.lr_scheduler` |

## Tenseurs et dtypes

Un tenseur a une forme, un dtype et un device. Un scalaire a la forme $()$, un vecteur $(n)$, une matrice $(m,n)$, un batch d'images $(B,C,H,W)$. Opérations clés : `reshape`/`view` (changer la forme sans copier les données ; `view` exige une mémoire contiguë), `.T`/`permute` (réordonner les axes), `@` (produit matriciel), `unsqueeze`/`squeeze`.

Deux faits mémoire à connaître issus de l'annexe du livre : `torch.tensor(array)` **copie** un tableau NumPy, alors que `torch.from_numpy(array)` **partage** la mémoire (modifier le tableau NumPy modifie le tenseur). Des listes d'entiers donnent des tenseurs entiers ; mélanger des tenseurs entiers et flottants dans une perte est une source classique d'erreurs (les étiquettes pour l'entropie croisée doivent être `long`).

Le dtype décide de la précision et de la plage. Valeurs mesurées avec `torch.finfo` :

| dtype | bits | epsilon machine | plus grande valeur | plus petit normal |
|---|---|---|---|---|
| float32 | 32 | $1.19\times10^{-7}$ | $3.4\times10^{38}$ | $1.18\times10^{-38}$ |
| float16 | 16 | $9.77\times10^{-4}$ | $65504$ | $6.1\times10^{-5}$ |
| bfloat16 | 16 | $7.8\times10^{-3}$ | $3.4\times10^{38}$ | $1.18\times10^{-38}$ |

Conséquences que tu peux vérifier toi-même : $70000$ en float16 devient `inf` ; $10^{-8}$ en float16 devient $0$ (il dépasse par le bas le plus petit sous-normal représentable) ; bfloat16 a la plage de float32 mais seulement 2 à 3 chiffres décimaux significatifs, donc $1+10^{-3}$ s'arrondit à $1$.

## Autograd

Chaque tenseur créé avec `requires_grad=True` est une feuille d'un graphe enregistré pendant la passe avant. `loss.backward()` le parcourt à l'envers et **ajoute** les gradients dans le `.grad` de chaque feuille. Trois règles en découlent :

1. Seules les feuilles avec `requires_grad=True` reçoivent `.grad`.
2. Les gradients s'accumulent. Trois appels de `(2*p).sum().backward()` sur un paramètre $p$ laissent `p.grad` égal à 6 au lieu de 2 (mesuré). C'est pourquoi la boucle efface les gradients à chaque pas ; l'accumulation est aussi l'astuce derrière l'accumulation de gradient pour de grands batches effectifs.
3. `torch.no_grad()` désactive l'enregistrement ; utilise-le pour l'évaluation et pour tout ce que tu ne veux pas dériver.

Le minuscule exemple du livre : $x=1.1$, $w=2.2$, $b=0$, cible $y=1$, $a=\sigma(wx+b)$ et entropie croisée binaire. PyTorch donne $a=0.9183$, perte $0.08519$, $\partial L/\partial w=-0.08983$, $\partial L/\partial b=-0.08166$. Cela concorde avec la dérivation à la main de [[f2n.10]] : pour sigmoïde plus BCE, $\partial L/\partial z=a-y=-0.08166$, puis $\partial L/\partial w=x\,(a-y)=1.1\times(-0.08166)=-0.08983$.

## nn.Module

```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, 30), nn.ReLU(),
            nn.Linear(30, 20), nn.ReLU(),
            nn.Linear(20, n_out),
        )
    def forward(self, x):          # x: (B, n_in) -> logits (B, n_out)
        return self.layers(x)
```

Affecter une couche comme attribut enregistre ses paramètres, donc `model.parameters()` les trouve récursivement. Pour `NeuralNetwork(50, 3)`, le nombre de paramètres est $50\cdot30+30+30\cdot20+20+20\cdot3+3=2213$ (le code du notebook `sum(p.numel() ...)` donne ce nombre ; la formule est poids plus biais par couche, comme pour le réseau NumPy). Le module porte aussi un mode : `model.train()` et `model.eval()` font basculer des couches comme le dropout ([[dl.02]]) et la batch norm ([[dl.03]]) entre leurs comportements d'entraînement et d'inférence. Les modèles produisent des **logits** ; le softmax se trouve à l'intérieur de `F.cross_entropy` (lui passer des probabilités donne silencieusement de mauvais gradients).

## Dataset et DataLoader

Un `Dataset` implémente `__len__` et `__getitem__` ; un `DataLoader` ajoute le découpage en batches, le mélange et des workers parallèles optionnels. Options de l'exemple du livre : `shuffle=True` pour l'entraînement uniquement, `drop_last=True` pour qu'un dernier batch court ne fausse pas les statistiques du batch, `num_workers` pour charger les données dans des processus d'arrière-plan (0 signifie dans le processus principal). Avec 5 points d'entraînement et une taille de batch de 2, le loader produit 3 batches (le dernier de taille 1), sauf si `drop_last` est activé, auquel cas il en produit 2.

## La boucle d'entraînement standard

```python
model = NeuralNetwork(2, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.5)
for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:            # xb: (B, 2), yb: (B,) long
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
```

Ligne par ligne par rapport à la boucle NumPy de [[f2n.11]] : `model(xb)` est la passe avant, `F.cross_entropy` la perte, `backward()` remplace ton `forward_backward` dérivé à la main, `opt.step()` est `sgd_step`, et `zero_grad()` est la remise à zéro dont ton code fonctionnel n'avait jamais besoin car il renvoyait de nouveaux gradients à chaque appel. Bugs typiques (du tableau de la leçon source) : oublier `zero_grad`, oublier `eval()` et `no_grad` au moment du test, désaccord CPU/GPU, mauvais dtype des étiquettes, garder le graphe en vie en stockant des pertes sans `.item()` ni `.detach()`, et opérations en place qu'autograd ne peut pas dériver.

Compléments de l'annexe D, qui ont leur place dans toute boucle sérieuse : un planning de taux d'apprentissage mis à jour à chaque pas ([[dl.01]] : warmup linéaire puis décroissance cosinus) et l'**écrêtage du gradient**, `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`, appelé après `backward()` et avant `step()`. Nous avons vérifié l'arithmétique du livre : pour un gradient d'entrées $(1,2;2,4)$, la norme est $\sqrt{25}=5$, et l'écrêtage à 1 le multiplie par $1/5$, donnant $(0.2,0.4;0.4,0.8)$.

## Devices et placement sur GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)                   # moves all parameters and buffers
xb, yb = xb.to(device), yb.to(device)      # move every batch
```

Tout ce qui se rencontre dans une même opération doit vivre sur le même device ; l'erreur « Expected all tensors to be on the same device » est l'échec de débutant le plus courant. Le modèle se déplace une fois, les données se déplacent à chaque batch. La fonction `compute_accuracy(model, loader, device)` du livre est exactement la fonction CPU avec deux appels `.to(device)` de plus. Copier des tenseurs entre CPU et GPU est lent, donc garde toute la boucle sur le device et ne rapatrie que des scalaires (avec `.item()`). L'entraînement multi-GPU avec DistributedDataParallel est couvert par le `DDP-script.py` du livre ; nous ne l'avons pas lu et ne le couvrons pas ici.

Sur cette machine, `torch.cuda.is_available()` est faux, et l'exercice s'exécute sur CPU.

## Précision mixte

Exécuter la passe avant et arrière en 16 bits divise par deux le trafic mémoire et, sur les GPU récents, augmente le débit, tandis que les paramètres et l'état de l'optimiseur restent en float32 (les « poids maîtres »). Deux variantes, directement issues du tableau des dtypes :

- **float16** a une plage étroite (plus grande valeur $65504$, plus petit normal $6\times10^{-5}$). Les petits gradients dépassent par le bas jusqu'à zéro, donc on multiplie la perte par un grand facteur avant `backward` et on divise ensuite : `GradScaler` automatise cela (la leçon source montre `autocast` plus `scaler.scale(loss).backward(); scaler.step(opt); scaler.update()`).
- **bfloat16** a la plage de float32 mais moins de bits de mantisse, donc il n'a pas besoin de mise à l'échelle de la perte ; c'est le choix courant pour les grands modèles de langage.

Sous `torch.autocast`, les multiplications matricielles s'exécutent en basse précision tandis que les opérations numériquement sensibles (softmax, réductions, pertes) sont gardées en float32 par PyTorch. Une vérification rapide sur CPU avec bfloat16 sur le modèle des demi-lunes entraîné : le tenseur de sortie a le dtype bfloat16 et les logits diffèrent de ceux en float32 d'au plus $0.085$ en valeur absolue. C'est un vrai coût de précision par nombre, acceptable pour l'entraînement, qui est robuste au bruit, et c'est la raison pour laquelle l'inférence en basse précision est évaluée avec soin.

## Points de sauvegarde (checkpointing)

`state_dict()` est un dictionnaire ordonné des noms de paramètres vers les tenseurs (pour le modèle des demi-lunes, les clés sont `0.weight, 0.bias, 2.weight, 2.bias` ; les indices sont les positions des couches dans le `Sequential`). Sauvegarde et chargement :

```python
torch.save(model.state_dict(), "model.pth")
model = NeuralNetwork(2, 2)                     # architecture must match exactly
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

`weights_only=True` restreint la désérialisation aux tenseurs, ce qui compte car un fichier pickle peut exécuter du code. Pour **reprendre** l'entraînement, tu dois aussi sauvegarder l'état de l'optimiseur (les $m$ et $v$ d'Adam et le compteur de pas de [[dl.01]]), l'epoch et le planning : `torch.save({"model":..., "opt":opt.state_dict(), "epoch":e}, path)`. Nous avons mesuré qu'un modèle rechargé depuis un point de sauvegarde en mémoire produit des sorties identiques bit à bit.

## Exemple résolu

L'exercice réutilise le réseau des demi-lunes et vérifie l'affirmation au cœur de cette leçon : PyTorch calcule la même chose que ton code NumPy.

- Copie les paramètres NumPy (graine 0) dans un `nn.Sequential(Linear(2,16), ReLU, Linear(16,2))` en float64, en transposant les poids.
- Perte : $0.60329\ldots$ dans les deux ; la différence est exactement $0.0$ en float64.
- Gradients : la différence absolue maximale sur les quatre tenseurs est $1.2\times10^{-16}$.
- Entraînement avec la boucle standard, initialisation de Kaiming, Adam ($\eta=0.03$), taille de batch 32, 60 epochs sur les 200 points : précision d'entraînement 1.0, perte $0.0128$, en environ 0.8 seconde sur CPU.

Si tu avais d'abord copié les poids dans un modèle float32 puis converti ensuite, l'accord ne serait que d'environ $10^{-9}$ (nous avons vu $3\times10^{-9}$) : la précision est perdue à la copie. Les tests de gradient se font en float64.

## Idées reçues

- **« PyTorch calcule quelque chose de différent de mon code NumPy. »** Les mêmes mathématiques ; l'exercice montre un accord à $10^{-16}$ en float64.
- **« Les poids de `nn.Linear` ont la forme $(n_{in},n_{out})$. »** Ils sont $(n_{out},n_{in})$ ; la couche calcule $xW^\top+b$.
- **« Seul le dropout a besoin de `eval()`. »** La batch norm utilise les statistiques mobiles en mode éval.
- **« `no_grad` et `eval()` sont la même chose. »** L'un arrête l'enregistrement du graphe, l'autre change le comportement des couches ; l'évaluation nécessite les deux.
- **« La précision mixte est gratuite. »** Elle échange de la précision contre de la vitesse et peut déborder ou dépasser par le bas (float16), c'est pourquoi la mise à l'échelle de la perte ou bfloat16 existent.
- **« Sauvegarder le modèle, c'est sauvegarder l'entraînement. »** `state_dict` ne contient que les paramètres (et les buffers) ; reprendre nécessite aussi l'optimiseur et le planning.

## Points clés à retenir

- Les tenseurs portent forme, dtype et device ; `from_numpy` partage la mémoire, `tensor` copie.
- Autograd enregistre la passe avant et la rejoue à l'envers ; les gradients s'accumulent, donc appelle `zero_grad` à chaque pas.
- Module, perte, optimiseur, DataLoader et la boucle en cinq lignes remplacent un à un les pièces que tu as écrites à la main.
- Déplace le modèle une fois et les données à chaque batch vers le même device ; utilise `eval()` plus `no_grad()` pour l'inférence.
- float16 nécessite une mise à l'échelle de la perte, bfloat16 non ; garde les poids maîtres en float32.
- Sauvegarde le `state_dict` avec `weights_only=True` au chargement, et sauvegarde l'état de l'optimiseur pour reprendre.

## Explique-le toi-même

1. Parcours une itération de la boucle PyTorch et nomme la fonction NumPy de [[f2n.11]] que chaque ligne remplace.
2. Pourquoi l'entraînement en float16 a-t-il besoin d'une échelle de perte mais pas celui en bfloat16 ? Utilise le tableau des dtypes.

## Pour aller plus loin

- *Build a Large Language Model From Scratch*, annexes A et D.
- Bibliothèque source : phase 03 leçon 11 (la construction MNIST en PyTorch pur y est décrite ; nous ne l'avons pas exécutée ici).

## Critères de maîtrise

Tu sais porter un réseau NumPy vers `nn.Module`, écrire les boucles d'entraînement et d'évaluation avec le bon mode et la bonne gestion des gradients, déplacer modèle et données vers un device, expliquer float16 contre bfloat16, sauvegarder et restaurer un point de sauvegarde y compris l'état de l'optimiseur, et vérifier le portage en comparant les gradients avec ton implémentation à partir de zéro.
