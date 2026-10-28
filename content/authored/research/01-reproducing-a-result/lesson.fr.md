---
id: res.01
title: "Reproduire un résultat publié de bout en bout"
summary: "Comment choisir une affirmation, extraire son dispositif exact, planifier le calcul, la reproduire à petite échelle sur de nombreuses graines, la comparer au chiffre publié avec des intervalles et des tolérances, documenter les écarts et diagnostiquer les échecs, avec une reproduction complète en NumPy de « le dropout réduit l'écart de généralisation »."
minutes: 85
teaches: [res.reproduction]
requires: [res.ablation, stat.testing, nn.training, dl.dropout]
sources:
  - "content/authored/llm-systems/05-research-practice/lesson.md (whole file): the three-pass reading, the short reproduction checklist and the paired-seed ablation that this lesson deepens without repeating. READ."
  - "content/authored/classical-ml/03-generalization/lesson.md (measuring protocol, experiment table) and content/authored/math/13-estimation-and-hypothesis-testing/lesson.md (Sec. Confidence intervals, Hypothesis tests, Comparing two models fairly): the Student interval, paired differences and multiple-comparison warnings used here. READ."
  - "content/authored/ml/11-training-from-scratch/lesson.md (Worked example: the half-moons): the moons problem and the idea of a small harness around a training loop; the model in this lesson is a variant (tanh, 64 hidden units, 60 training points) chosen so that it overfits. READ."
  - "reasoning-from-scratch/ch03/01_main-chapter-code/ch03_main.ipynb (markdown results table): the same 10-problem evaluation of the base model gives 30 percent on CPU, CUDA and XPU but 20 percent on MPS, an example of hardware-dependent results. READ."
  - "reasoning-from-scratch/ch04/01_main-chapter-code/ch04_main.ipynb (code cells with torch.manual_seed(123) before every sampling demo; self_consistency_vote(seed=...) which seeds each sample as seed+i+1; markdown on stochastic decoding): how a stochastic procedure is made repeatable only by seeding it explicitly. READ."
  - "Designing Machine Learning Systems.pdf, Ch.6 (PDF p.190-191 Experiment Tracking and Versioning; p.201-205 Model Offline Evaluation and Baselines): artifacts, versioning code and data so that a promising run can be recreated, and why a number needs a baseline to mean anything. READ."
---

## Où cela s'insère

Un article dit « la méthode X améliore la métrique Y de 2 points ». Avant de construire dessus, tu voudrais savoir si c'est vrai sur ton matériel, avec ton code. La reproduction est la façon de transformer la phrase de quelqu'un d'autre en une preuve qui t'appartient. C'est aussi la meilleure façon d'apprendre une méthode : on ne peut pas reproduire ce qu'on ne comprend pas, et les endroits où la reproduction diverge de l'article sont ceux où se cachent les leçons intéressantes.

[[sys.05]] t'a donné la version courte : lire l'article en trois passes, reproduire d'abord la ligne de base, s'attendre à de petits écarts, et mener une ablation appariée. Cette leçon approfondit la partie que celle-là se contentait de lister. Comment choisir une affirmation qui vaut la peine d'être reproduite ? Comment extraire le dispositif *complet* d'un article qui en omet la moitié ? Comment décider, chiffres à l'appui, que ton résultat et celui de l'article concordent ? Que consigner pour que la personne suivante (en général toi, dans six mois) puisse se fier à ton verdict ? Nous utilisons les outils statistiques de [[math.13]] (intervalles, différences appariées), le protocole de généralisation de [[cml.03]] et la boucle d'entraînement de [[f2n.11]].

Après cette leçon, la suite [[res.02]] retourne la même mécanique : au lieu de vérifier l'affirmation de quelqu'un d'autre, tu conçois ta propre expérience de sorte qu'elle puisse échouer honnêtement.

## L'idée en mots simples

Imagine qu'un ami te donne une recette de gâteau en disant « il gonfle toujours magnifiquement ». Tu la suis une fois et le gâteau est plat. Qu'est-ce que cela t'apprend ? Peut-être que la recette est mauvaise. Peut-être que ton four chauffe peu. Peut-être que la recette a oublié de dire que les œufs doivent être à température ambiante. Peut-être que tu as simplement eu de la malchance cette fois. Pour le savoir, tu ne cuis pas dix fois de plus le même gâteau plat ; tu cuis plusieurs gâteaux, tu notes tout ce que tu as fait, tu changes une chose à la fois, et tu compares la *dispersion* de tes résultats à ce que l'ami a promis.

Reproduire un résultat, c'est exactement cela, avec cinq habitudes :

- **Choisis une affirmation, pas un article.** Un article contient de nombreuses affirmations. Tu en reproduis une phrase avec un chiffre ou une direction (« A bat B »), parce que c'est une question qui a une réponse.
- **Écris la recette avant de cuisiner.** Données, modèle, hyperparamètres, métrique, nombre de graines, matériel. Tout ce que tu ne trouves pas dans l'article devient un « j'ai supposé ceci » explicite.
- **Commence petit.** Une version réduite qui devrait montrer le même effet *qualitatif* est assez bon marché pour être lancée de nombreuses fois.
- **Compare des plages, pas des points.** Tes dix exécutions donnent une plage. Le chiffre unique de l'article est un tirage de sa propre plage. Ils concordent quand le chiffre de l'article est plausible sous ta plage, pas quand les deux décimales coïncident.
- **Note ce qui diffère.** Une reproduction qui finit par « assez proche » sans liste d'écarts n'apprend rien à personne.

## Qu'est-ce qui compte comme « reproduit » ? Un vocabulaire

Les chercheurs emploient plusieurs mots à la légère ; fixe-les avant de commencer :

| Terme | Même code | Mêmes données | Question à laquelle on répond |
|---|---|---|---|
| Ré-exécution (répétabilité) | oui | oui | Le script d'origine donne-t-il le même chiffre deux fois sur la même machine ? |
| Reproduction | oui (le leur) | oui | Leur code et leurs données donnent-ils leur chiffre sur ma machine ? |
| Réplication | non (le mien) | oui ou nouvelles | La *découverte* survit-elle à une implémentation indépendante ? |
| Généralisation de la découverte | non | différentes | L'affirmation tient-elle sur d'autres données ou à une autre échelle ? |

Cette leçon porte sur les deuxième et troisième lignes, exécutées délibérément à petite échelle. Une reproduction à petite échelle ne peut généralement soutenir qu'un verdict **qualitatif** (« l'écart se réduit », « A converge plus vite que B ») et un verdict **quantitatif** seulement si tu peux coller de près au dispositif. Le dire dans le rapport fait partie du travail.

## Étape 1 : choisir l'affirmation

De bonnes affirmations pour une première reproduction ont quatre propriétés :

1. **Précise.** « Un dropout de 0.5 réduit l'écart de précision entre entraînement et test d'un réseau à une couche cachée » plutôt que « le dropout aide ».
2. **Réfutable à ton échelle.** Si elle exige 1 000 heures de GPU, réduis la question (modèle plus petit, sous-ensemble des données) ou choisis une autre affirmation.
3. **Mesurable avec une métrique calculable exactement.** Écart entraînement-test, précision, perte à l'étape $k$, étapes pour atteindre un seuil.
4. **Intéressante si fausse.** Certaines affirmations sont du folklore sur lequel tout le monde s'appuie ; les confirmer est peu coûteux et précieux.

Écris l'affirmation sous la forme d'une phrase du type *« Dans le cadre S, la configuration A a une métrique M (plus basse/plus haute) que la configuration B, d'environ Δ. »* La partie « d'environ Δ » est facultative mais transforme une direction vague en chiffre testable.

## Étape 2 : extraire le dispositif exact

Lis la méthode, les expériences et l'annexe de l'article avec un tableau ouvert et remplis chaque ligne. Les lignes qui restent vides sont tes *détails non rapportés*.

| Champ | Que consigner | Lacune typique dans les articles |
|---|---|---|
| Données | source, version, taille, prétraitement, tailles des découpes, qui a fait la découpe | découpe exacte ou règle de filtrage non précisée |
| Modèle | architecture, taille, initialisation | schéma d'initialisation ou placement de la normalisation non précisé |
| Entraînement | optimiseur, planification du taux d'apprentissage, taille de lot, étapes ou epochs, régularisation | durée du warmup, exclusions du weight decay |
| Métrique | définition exacte, moyennage, sur quelle découpe elle est mesurée | moyenne macro vs micro, seuil réglé |
| Graines | combien, ce que contrôle chacune | généralement une, ou non mentionné |
| Calcul | matériel, précision, versions de bibliothèques, temps réel | rarement donné |
| Lignes de base | lesquelles, comment elles ont été réglées | ligne de base moins réglée que la méthode proposée |

Pour les détails introuvables, par ordre de préférence : cherche dans le code publié, puis l'annexe, puis les travaux voisins des mêmes auteurs, et enfin choisis une valeur par défaut et **consigne-la comme écart**. Le chapitre de Huyen sur le suivi d'expériences (Ch.6, PDF p.190-191) fait le même constat côté ingénierie : conserve toutes les définitions nécessaires pour recréer une expérience et ses artefacts (journaux, courbes, résultats intermédiaires), et versionne le code *et* les données, car une exécution prometteuse qu'on ne peut pas recréer ne vaut rien.

## Étape 3 : planifier le calcul

Estime le coût avant de t'engager. Si une exécution prend $c$ minutes, que le plan comporte $R$ exécutions et que ton budget est $B$ minutes, il te faut $R\,c \le B$. Les exécutions sont

$$
R = (\text{configurations}) \times (\text{seeds}) .
$$

Pour une comparaison à deux bras avec 10 graines, cela fait 20 exécutions. Si l'exécution à pleine échelle est hors de portée, réduis **une** chose à la fois et note-le : moins d'étapes, une largeur plus petite, un sous-ensemble de données. La réduction d'échelle est elle-même un écart, car certains effets (par exemple le comportement émergent des grands modèles) disparaissent quand on rétrécit. Un contrôle de bon sens utile est de lancer d'abord le plus petit cas et de vérifier qu'il produit *quelque chose* qui ressemble à la tendance publiée avant d'investir dans plus de graines.

Décide aussi dès maintenant ce que tu feras de la réponse. Si l'affirmation est reproduite, tu construis dessus ; sinon, tu passeras du temps à diagnostiquer. Fixe une limite de temps au diagnostic, sans quoi une reproduction ratée avale une semaine.

## Étape 4 : comparer au chiffre de l'article

Supposons que l'article rapporte un nombre $r$ et que tu obtiennes $n$ exécutions $x_1,\dots,x_n$ de moyenne d'échantillon $\bar x$ et d'écart type d'échantillon $s$ (avec $n-1$ au dénominateur). L'intervalle de Student à 95 % de [[math.13]] est

$$
\bar x \pm t_{n-1,\,0.975}\,\frac{s}{\sqrt n}.
$$

Trois issues méritent d'être distinguées :

1. **Reproduit** : $r$ se trouve dans l'intervalle. Le chiffre de l'article est un tirage plausible du processus que tu as mesuré.
2. **Dans la tolérance** : $r$ est hors de l'intervalle mais $|\bar x - r| \le \tau$ où $\tau$ est une tolérance fixée *avant* l'exécution (par exemple « à un point de pourcentage près », ou « dans l'écart type rapporté par l'article lui-même »). Dis que c'est dans la tolérance, pas que c'est reproduit.
3. **Non reproduit** : ni l'un ni l'autre. C'est une découverte, pas un échec, tant que tu peux dire ce que tu as vérifié.

Deux précautions. Premièrement, l'intervalle décrit l'incertitude de **ta moyenne** ; la dispersion des exécutions individuelles est $s$, soit $\sqrt n$ fois plus large. Si l'article rapporte *une* exécution, la bonne comparaison pour « l'exécution de l'article est-elle typique ? » se fait avec $\bar x \pm 2s$, pas avec l'intervalle étroit de la moyenne. Deuxièmement, un intervalle large n'est pas une reproduction. Avec $n=3$ la valeur critique est $4.30$ et presque n'importe quel $r$ tombe dedans ; « reproduit » n'a de sens que si l'intervalle est assez étroit pour exclure quelque chose. Rapporte toujours l'intervalle avec le verdict.

Pour une affirmation **qualitative** (« A est meilleur que B ») compare les deux bras avec la différence appariée. Lance les deux bras avec les mêmes graines et pose $d_i = a_i - b_i$ les différences par graine. L'intervalle de la différence moyenne est $\bar d \pm t_{n-1,0.975}\, s_d/\sqrt n$. S'il exclut 0, la direction est soutenue dans ton cadre ; sa largeur te dit l'ampleur de l'effet. C'est la statistique appariée de [[sys.05]] ; ici nous l'utilisons comme *verdict* de la reproduction.

## Pourquoi les reproductions échouent

Quand ton chiffre diverge de celui de l'article, parcours cette liste dans l'ordre, la vérification la moins chère d'abord.

1. **Détails non rapportés.** Initialisation, warmup, weight decay sur les termes de biais, filtrage des données, astuces d'évaluation (augmentation au moment du test, sélection de checkpoint). Cherche dans le code publié ; essaie les valeurs par défaut évidentes ; consigne chacune comme écart.
2. **Évaluation différente.** La métrique, la découpe, le moyennage ou le seuil diffère. Recalcule la *ligne de base* avec ton code d'évaluation ; si tu ne peux pas retrouver le chiffre publié de la ligne de base, l'évaluation est suspecte, et aucune comparaison avec la méthode proposée n'est fiable.
3. **Fuite de données dans l'original ou chez toi.** Si l'article a sélectionné des hyperparamètres ou des checkpoints sur le jeu de test, son chiffre est optimiste ([[cml.03]] : on ne touche le jeu de test qu'une fois). Ton propre pipeline peut aussi fuir : normaliser avec des statistiques calculées sur tout le jeu de données, ou augmenter avant de découper. Une reproduction qui sort *meilleure* que l'article est aussi suspecte qu'une qui sort moins bonne.
4. **Graines et variance.** L'article peut rapporter la meilleure de plusieurs exécutions, ou une exécution chanceuse. Si son chiffre est dans la queue de ta distribution, « non reproduit » peut vouloir dire « rapporté à partir d'un tirage favorable ». Les graines peuvent contrôler les poids initiaux, l'ordre des lots, les masques de dropout et la découpe des données ; dis lesquelles tu fais varier. (Le code des carnets reasoning-from-scratch appelle `torch.manual_seed(123)` avant chaque démonstration d'échantillonnage, et sa routine d'auto-cohérence initialise le $i$-ème échantillon avec `seed + i + 1` : une procédure stochastique n'est répétable que parce que quelqu'un l'a initialisée délibérément.)
5. **Non-déterminisme au-delà des graines.** Le même code sur un autre matériel ou d'autres versions de bibliothèques peut donner des chiffres différents. Le tableau de résultats du chapitre reasoning-from-scratch sur l'évaluation montre un modèle de base noté sur les mêmes 10 problèmes : 30 % sur CPU, CUDA et XPU, mais 20 % sur le backend MPS d'Apple. Avec seulement 10 problèmes, un problème fait basculer le chiffre de dix points, donc la différence mêle non-déterminisme numérique et minuscule taille d'échantillon ; la leçon est que « le même code » ne garantit pas « le même chiffre ». Sources typiques : ordre de sommation en virgule flottante sur GPU, noyaux non déterministes, précision par défaut différente, versions de bibliothèques différentes.
6. **Échelle.** L'effet existe à l'échelle de l'article et disparaît à la tienne (ou l'inverse). C'est le seul échec qu'on ne peut pas corriger en déboguant ; tu le rapportes comme une limite de la reproduction.
7. **L'affirmation était plus faible que son résumé.** Relis le tableau. Souvent le titre dit « améliore de 2 points » et le tableau montre 2 points sur un jeu de données avec des barres d'erreur qui se chevauchent.

Décide d'une règle d'arrêt à l'avance : par exemple, « après avoir épuisé les points 1 à 5 et une journée d'effort, j'arrête et je rapporte "non reproduit, voici ce que j'ai essayé" ».

## Étape 5 : documenter les écarts et écrire le rapport

Chaque différence entre ton dispositif et celui de l'article est un **écart**, et un rapport de reproduction est principalement un tableau d'écarts. Un rapport complet comporte ces sections (c'est exactement ce que cherche le vérificateur du second exercice) :

```text
claim            one sentence: setting, A vs B, metric, direction and size
source           paper, table or figure, version, code release used or not
dataset          source, sizes, splits, preprocessing
model            architecture and size
hyperparameters  every value, and which were tuned for which arm
metric           exact definition and the split it is measured on
seeds            list of seeds, and what each seed controls
environment      hardware, library versions, precision, run time
deviations       every difference from the paper, or the word "none"
results          per-arm mean, sample sd, 95 percent interval, n, paired difference
verdict          reproduced / within tolerance / not reproduced / partly, in one sentence
limits           what the result does NOT show (scale, data, untested settings)
artifacts        commit hash, command to rerun, log files
```

Règles pour la phrase du verdict : dis ce que tu as observé en chiffres, nomme l'intervalle, sépare l'observation de l'interprétation, et n'emploie jamais un mot plus fort que ce que les preuves soutiennent. « Partiellement reproduit : l'écart se réduit avec le dropout dans le cadre 1 (10 graines sur 10), mais nous n'avons pas pu le montrer dans le cadre 2 » est une conclusion légitime et utile.

## Exemple travaillé : le dropout réduit-il l'écart de généralisation ?

**L'affirmation (de niveau manuel).** « Sur un petit jeu de données bruité, un réseau avec un dropout de 0.5 sur sa couche cachée a un écart plus petit entre la précision d'entraînement et la précision de test que le même réseau sans dropout. » (Le dropout a été couvert dans [[dl.02]] ; ici nous le traitons comme l'affirmation de quelqu'un d'autre à vérifier.)

**Dispositif, extrait avant l'exécution.**

- Données : deux lunes bruitées ([[f2n.11]]), écart type du bruit 0.3, **60** points d'entraînement (petit exprès, pour que le réseau puisse surapprendre) et 2 000 points de test. La graine de données 1234 fixe un jeu de données dans le cadre 1.
- Modèle : $2 \to 64 \to 1$, unités cachées tanh, sortie logistique, poids $W_1\sim\mathcal N(0,1)$, $W_2\sim\mathcal N(0,1/64)$, biais nuls.
- Entraînement : descente de gradient sur le lot complet, taux d'apprentissage 0.1, 1 500 étapes, perte d'entropie croisée. Le dropout (inversé, taux 0.5) est appliqué aux activations cachées pendant l'entraînement seulement, avec un masque neuf à chaque étape.
- Métrique : écart $=$ précision d'entraînement moins précision de test, toutes deux mesurées sans dropout. La graine contrôle les poids initiaux et les masques de dropout.
- Plan : 10 graines (0 à 9), les deux bras avec les mêmes graines (nombres aléatoires communs), 20 exécutions, chacune prend environ 0.4 s, donc le budget est trivial. Tolérance pour un chiffre rapporté hypothétique : 0.02.

```python
import numpy as np

def make_moons(n, noise, rng):
    t = rng.uniform(0, np.pi, n // 2)
    a = np.c_[np.cos(t), np.sin(t)]
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.r_[a, b] + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return X, y

def train_mlp(seed, p_drop=0.0, hidden=64, steps=1500, lr=0.1,
              n_train=60, noise=0.3, data_seed=1234):
    drng = np.random.default_rng(data_seed)
    Xtr, ytr = make_moons(n_train, noise, drng)
    Xte, yte = make_moons(2000, noise, drng)
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0, (2, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, 1)); b2 = np.zeros(1)
    n = len(Xtr)
    for _ in range(steps):
        H = np.tanh(Xtr @ W1 + b1)                      # (n, hidden)
        if p_drop > 0:
            mask = (rng.random(H.shape) >= p_drop) / (1 - p_drop)
        else:
            mask = 1.0
        Hd = H * mask
        p = 1 / (1 + np.exp(-(Hd @ W2 + b2)[:, 0]))
        dz = ((p - ytr) / n)[:, None]                   # (n, 1)
        gW2 = Hd.T @ dz; gb2 = dz.sum(0)
        dH = (dz @ W2.T) * mask * (1 - H ** 2)          # (n, hidden)
        gW1 = Xtr.T @ dH; gb1 = dH.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    def acc(X, y):
        H = np.tanh(X @ W1 + b1)
        return float((((H @ W2 + b2)[:, 0] > 0) == (y == 1)).mean())
    return acc(Xtr, ytr), acc(Xte, yte)             # (train acc, test acc)
```

Avant toute statistique, lance le **contrôle de déterminisme** : `train_mlp(3, 0.5) == train_mlp(3, 0.5)` vaut `True` ici (tuple identique sur la même machine). Sans cela, les différences entre bras pourraient être du bruit de la machine plutôt que de la méthode.

**Cadre 1 : un jeu de données fixe, dix graines.** Les graines font varier les poids initiaux et les masques de dropout. Résultats (moyenne sur les graines, intervalle à 95 % entre crochets) :

| Quantité | Sans dropout | Dropout 0.5 |
|---|---|---|
| Précision d'entraînement | 0.962 | 0.888 |
| Précision de test | 0.891 | 0.873 |
| Écart (entraînement $-$ test) | 0.0708 [0.0614, 0.0801] | 0.0157 [0.0088, 0.0226] |

Apparié par graine, la réduction de l'écart est de $0.0551$ avec un intervalle à 95 % de $[0.0447,\ 0.0655]$ (t apparié : $t = 11.95$ sur 9 degrés de liberté) et le dropout a le plus petit écart dans **10 graines sur 10**. La précision de test change de $-0.0183$, intervalle $[-0.0210,\ -0.0155]$, et le dropout est meilleur dans **0 graine sur 10**.

Supposons qu'un article ait rapporté un écart de 0.020 avec dropout. Cette valeur se trouve dans $[0.0088, 0.0226]$, donc selon la règle ci-dessus le verdict sur ce chiffre est « reproduit ». (Cette valeur rapportée est hypothétique ; l'enjeu est la façon de comparer.)

**Cadre 2 : changer ce que contrôle la graine.** Le cadre 1 gardait le jeu de données fixe, donc ses intervalles ne contiennent que le bruit d'initialisation et de dropout. Un lecteur sceptique demandera ce qui se passe quand le *jeu d'entraînement lui-même* est retiré. Répète avec `data_seed = 100 + seed`, de sorte que chaque graine reçoit aussi un nouveau jeu d'entraînement de 60 points et un nouveau jeu de test :

| Quantité | Valeur |
|---|---|
| Écart, sans dropout | 0.0589 (sd 0.041) |
| Écart, dropout 0.5 | 0.0508 (sd 0.052) |
| Réduction de l'écart, appariée | 0.0082, intervalle à 95 % $[-0.0104,\ 0.0267]$ |
| Variation de la précision de test | $-0.0168$, intervalle $[-0.0238,\ -0.0099]$, dropout meilleur dans 0 sur 10 |

Maintenant l'intervalle de la réduction de l'écart contient zéro. Les différences appariées sont dispersées (de $-0.025$ à $+0.055$) parce qu'un seul jeu de 60 points peut surapprendre plus ou moins quoi qu'on fasse.

**Verdict écrit.** *Partiellement reproduit.* Sur un jeu fixe de 60 points, le dropout 0.5 a réduit l'écart de précision entraînement-test de 0.071 à 0.016 (différence appariée 0.055, intervalle à 95 % de 0.045 à 0.066, 10 graines sur 10). Quand le jeu de données est aussi rééchantillonné, la réduction tombe à 0.008 et son intervalle $[-0.010, 0.027]$ inclut zéro avec 10 graines ; pour voir un effet de cette taille avec une puissance de 80 % il faudrait environ 80 exécutions appariées (le calcul de puissance est établi dans [[res.02]]). Dans les deux cadres, le dropout a abaissé la précision de test (d'environ 0.017 à 0.018) dans chaque graine, donc l'écart plus étroit venait d'une précision d'entraînement plus basse, pas d'une meilleure généralisation. Écarts : implémentation NumPy maison, réseau tanh, mises à jour sur lot complet, aucun réglage du taux de dropout ni du taux d'apprentissage pour l'un ou l'autre bras. Limites : une architecture, une famille de jeux de données et 1 500 étapes ; on n'a pas testé si le dropout aiderait avec un calendrier plus long ou une ligne de base réglée.

Remarque ce que le rapport ne dit *pas* : il ne dit pas que le dropout « marche » ou « ne marche pas ». Un écart plus étroit est l'affirmation, et dans le cadre 1 cette affirmation tient ; savoir si un écart plus étroit est utile est une question distincte à laquelle les mêmes chiffres répondent par la négative.

## Implémentation : le harnais

La même structure est le premier exercice. Le code ci-dessous est tout le harnais.

```python
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228]  # df 1..10 (extend)

def run_seeds(fn, seeds):
    return np.array([fn(s) for s in seeds], dtype=float)

def summarize(x):
    x = np.asarray(x, dtype=float); n = len(x)
    half = T975[n - 2] * x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": x.mean(), "sd": x.std(ddof=1), "lo": x.mean() - half, "hi": x.mean() + half}

gaps_a = run_seeds(lambda s: np.subtract(*train_mlp(s, 0.0)), range(10))   # train - test
gaps_b = run_seeds(lambda s: np.subtract(*train_mlp(s, 0.5)), range(10))
print(summarize(gaps_a - gaps_b))      # paired difference: same seeds in both arms
```

Dans du code de recherche de production, tu utiliserais les outils de suivi d'expériences mentionnés par Huyen (par exemple MLflow ou Weights and Biases) pour stocker automatiquement ces tableaux, configurations et hachages de commits ; la version minimale est un fichier JSON par exécution contenant la configuration, la graine, l'environnement et les métriques.

## Formes

| Objet | Forme |
|---|---|
| entrées / étiquettes d'entraînement | $(60, 2)$ / $(60,)$ |
| $W_1$, $b_1$, $W_2$, $b_2$ | $(2, 64)$, $(64,)$, $(64, 1)$, $(1,)$ |
| masque de dropout, activations cachées | $(60, 64)$ chacun |
| résultats d'un bras (entraînement, test) | $(n_\text{seeds}, 2)$ |
| différences appariées | $(n_\text{seeds},)$ |

## Idées reçues

1. « Si mon chiffre colle à celui de l'article à deux décimales, je l'ai reproduit. » Ce peut être de la chance ; une concordance dans un large intervalle apporte peu d'information. Rapporte l'intervalle et le nombre d'exécutions.
2. « Si mon chiffre est en dessous de celui de l'article, l'article a tort. » Vérifie d'abord l'évaluation, la ligne de base, la distribution des graines et les détails non rapportés. Les auteurs peuvent aussi rapporter des exécutions favorables sans mauvaise intention.
3. « Une seule graine reproduit un résultat si le code est le même. » Le code fixe la procédure, pas le tirage aléatoire. Une seule graine montre un échantillon de la distribution.
4. « Initialiser la graine rend l'exécution déterministe. » La graine fixe les flux pseudo-aléatoires que tu contrôles ; l'ordre en virgule flottante sur matériel parallèle, les versions de bibliothèques et les noyaux non déterministes peuvent encore changer les résultats.
5. « Une reproduction ratée est une semaine perdue. » Un échec bien documenté (ce que tu as essayé, quels étaient les intervalles) est un résultat ; le gaspillage est un échec non documenté.
6. « Un accord qualitatif signifie que la taille d'effet de l'article est juste. » Dans l'exemple travaillé, le signe de l'effet a été reproduit alors que la conclusion pratique (le modèle est-il meilleur ?) était l'opposé de ce que suggère l'écart seul.

## Points clés à retenir

- Choisis une affirmation précise ; écris-la comme une phrase avec cadre, bras, métrique et direction.
- Extrais le dispositif dans un tableau ; chaque ligne vide est un détail non rapporté que tu dois remplir et consigner comme écart.
- Planifie les exécutions comme configurations fois graines et vérifie le budget avant de commencer.
- Compare à l'article avec un intervalle sur tes propres exécutions et une tolérance déclarée à l'avance ; utilise des différences appariées pour « A bat B ».
- Vérifie le déterminisme et reproduis d'abord la ligne de base ; étudie les échecs dans un ordre fixe et avec une règle d'arrêt.
- Précise quelles sources de variance tes graines couvrent ; les élargir (nouvelles données, pas seulement nouvelle initialisation) peut changer le verdict.
- Un rapport est un tableau d'écarts plus une phrase de verdict qui n'affirme pas plus que les preuves.

## Explique-le toi-même

1. Dans le cadre 1, l'intervalle de la réduction de l'écart était $[0.045, 0.066]$ ; dans le cadre 2, il était $[-0.010, 0.027]$. Les deux utilisent la même méthode et les mêmes dix graines. Explique ce qui a changé et quelle expérience soutient le mieux une affirmation sur « le dropout en général ».
2. Un collègue reproduit un article avec une exécution et écrit « reproduit, 71.2 contre 71.5 ». Écris les trois choses que tu demanderais avant d'accepter cette phrase.

## Pour aller plus loin

- Henderson et al., « Deep Reinforcement Learning that Matters » (2018) : échecs de reproductibilité dus aux graines et aux détails d'implémentation.
- Bouthillier et al., « Accounting for Variance in Machine Learning Benchmarks » (2021) : quelles sources de variance randomiser.
- Pineau et al., « Improving Reproducibility in Machine Learning Research » (2020) : listes de contrôle pour les articles.
- Huyen, *Designing Machine Learning Systems*, Ch. 6 : suivi d'expériences et versionnage.

## Critères de maîtrise

- [ ] Je sais transformer une phrase d'article en une affirmation précise et réfutable et remplir un tableau de dispositif en marquant les détails non rapportés.
- [ ] Je sais budgéter une reproduction comme configurations fois graines fois minutes par exécution.
- [ ] Je sais comparer mes exécutions à un chiffre rapporté avec un intervalle de Student et une tolérance fixée à l'avance.
- [ ] Je sais dire quelles sources de variance une graine contrôle et les élargir délibérément.
- [ ] Je sais lancer un contrôle de déterminisme et lister les causes d'échec dans l'ordre où je les étudierais.
- [ ] Je sais écrire un rapport de reproduction dont le verdict n'affirme pas plus que les preuves.
