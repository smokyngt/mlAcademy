---
id: nlp.03
title: "Étiquetage de séquences avec les HMM et Viterbi"
summary: "Étiquette chaque mot avec une catégorie grammaticale ou une étiquette d'entité à l'aide d'un modèle de Markov caché, dérive le programme dynamique de Viterbi, implémente-le, et vois pourquoi les CRF sont venus ensuite."
minutes: 80
teaches: [nlp.hmm-viterbi]
requires: [prob.basics, alg.exp-log]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/06-named-entity-recognition/docs/en.md: BIO tagging, progression from rules to HMM to CRF to neural taggers. READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing/docs/en.md: POS tags, most-frequent-tag baseline, constituency versus dependency parsing. READ."
  - "math/cours-tal-01.pdf (PDF p.12-26, French slides): Markov chains, hidden Markov models for POS tagging, naive search over tag sequences, Viterbi dynamic programming, use of logs and the smoothing problem. READ."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 13.2 Hidden Markov Models (PDF p.630): the joint distribution of an HMM and the conditional-independence argument. Only the first page was read. READ."
---

## Où ça s'inscrit

La classification ([[nlp.02]]) donne une étiquette par document. De nombreuses tâches de langage demandent une étiquette par **mot** : quels mots sont des noms ou des verbes (l'étiquetage morpho-syntaxique, nécessaire au lemmatiseur de [[nlp.01]]), quels mots appartiennent à une personne, un lieu ou une organisation (la reconnaissance d'entités nommées, NER, première étape de l'extraction d'information). Les étiquettes de mots voisins dépendent les unes des autres : après « the » un nom est probable, après « to » un verbe. Un modèle qui étiquette les mots indépendamment ignore cela. Le **modèle de Markov caché** (HMM) le capture avec une histoire probabiliste simple, et l'**algorithme de Viterbi** trouve la meilleure séquence d'étiquettes parmi un nombre exponentiel en temps linéaire. Les étiqueteurs neuronaux ([[seq.03]] sur les réseaux récurrents, et les transformers) remplacent aujourd'hui le HMM, mais les étiquettes BIO, le décodage par programmation dynamique et l'idée des CRF y survivent tous.

## L'idée en termes simples

Imagine une personne qui écrit des phrases en parcourant un petit nombre de pièces, une par catégorie grammaticale : DET, NOUN, VERB. Dans chaque pièce elle dit un mot, puis elle va à la pièce suivante, choisie selon l'habitude : depuis DET on va presque toujours vers NOUN. Tu n'entends que les mots, pas les pièces. Étant donnés les mots « the fish bark », tu veux le parcours le plus plausible. « Fish » peut être dit dans la pièce NOUN ou dans la pièce VERB, donc tu ne peux pas décider mot par mot ; tu dois peser le parcours entier, où les habitudes de transition aident (« DET -> NOUN » est probable).

Vérifier tous les parcours possibles est sans espoir pour de longues phrases (3 pièces et 20 mots font déjà plus de trois milliards de parcours). L'astuce derrière Viterbi : si deux parcours partiels arrivent dans la même pièce au même moment, seul le meilleur peut jamais faire partie du meilleur parcours complet ; on garde donc un champion par pièce et par pas.

## Modèle

Un HMM d'ordre un a des états cachés $t_1,\dots,t_n$ (les étiquettes) et des mots observés $x_1,\dots,x_n$. Trois tables de paramètres, pour $K$ étiquettes et une taille de vocabulaire $V$ :

- probabilités initiales $\pi_k=P(t_1=k)$, de forme $(K,)$ ;
- probabilités de transition $A_{jk}=P(t_i=k\mid t_{i-1}=j)$, de forme $(K,K)$, chaque ligne a pour somme 1 ;
- probabilités d'émission $B_{kw}=P(x_i=w\mid t_i=k)$, de forme $(K,V)$, chaque ligne a pour somme 1.

Deux hypothèses : chaque étiquette ne dépend que de l'étiquette précédente (propriété de Markov), et chaque mot ne dépend que de sa propre étiquette. La probabilité jointe d'un étiquetage est donc

$$
P(x,t)=\pi_{t_1}B_{t_1x_1}\prod_{i=2}^{n}A_{t_{i-1}t_i}B_{t_ix_i}.
$$

C'est la factorisation donnée pour le HMM dans Bishop, Sec. 13.2, et dans les diapositives. Étiqueter, c'est trouver $\arg\max_t P(t\mid x)$ ; comme $P(t\mid x)=P(x,t)/P(x)$ et que $P(x)$ ne dépend pas de $t$, nous maximisons la probabilité jointe.

**Apprendre à partir de texte étiqueté.** Avec un corpus étiqueté, le maximum de vraisemblance est du comptage : $\pi_k$ = fraction des phrases commençant par l'étiquette $k$ ; $A_{jk}=\frac{c(j\to k)}{c(j)}$ ; $B_{kw}=\frac{c(k,w)}{c(k)}$. Comme dans [[nlp.02]], ajoute un pseudo-compte $\alpha$ pour que les événements non vus n'aient pas une probabilité nulle : les diapositives signalent exactement ce problème de lissage, ainsi que celui des mots inconnus (un remède usuel est un mot conditionnel à l'étiquette `<UNK>` ou des caractéristiques de suffixe).

## Dérivation : Viterbi

Naïvement il y a $K^n$ séquences d'étiquettes. Définis

$$
\delta_i(k)=\max_{t_1,\dots,t_{i-1}}P(x_{1:i},\,t_{1:i-1},\,t_i=k),
$$

la probabilité du meilleur étiquetage partiel des $i$ premiers mots se terminant par l'étiquette $k$.

**Cas de base.** $\delta_1(k)=\pi_kB_{kx_1}$.

**Récurrence.** Un étiquetage partiel se terminant par $k$ à la position $i$ consiste en un étiquetage partiel se terminant par un certain $j$ à la position $i-1$, puis le facteur $A_{jk}B_{kx_i}$. Le nouveau facteur dépend du passé uniquement via $j$ (propriété de Markov), donc pour un $j$ fixé le meilleur choix des étiquettes précédentes est celui qui maximise $\delta_{i-1}(j)$. Par conséquent

$$
\delta_i(k)=\Big[\max_{j}\ \delta_{i-1}(j)\,A_{jk}\Big]\,B_{kx_i},\qquad \psi_i(k)=\arg\max_j\ \delta_{i-1}(j)A_{jk},
$$

où le pointeur arrière $\psi_i(k)$ enregistre quelle étiquette précédente a gagné.

**Terminaison et remontée.** La meilleure étiquette finale est $t_n^*=\arg\max_k\delta_n(k)$ ; puis $t_{i-1}^*=\psi_i(t_i^*)$ pour $i=n,\dots,2$.

**Coût.** Chacune des $n\cdot K$ cases regarde $K$ prédécesseurs : $O(nK^2)$ au lieu de $O(K^n)$.

**Logs.** Multiplier de nombreuses probabilités provoque un underflow, donc stocke $\log\delta$ et remplace les produits par des sommes : $\log\delta_i(k)=\max_j[\log\delta_{i-1}(j)+\log A_{jk}]+\log B_{kx_i}$. Une probabilité nulle devient $-\infty$, ce qui est sans danger pour le maximum.

Remplacer le max par une somme donne l'*algorithme forward*, qui calcule $P(x)$ ; il n'est pas nécessaire ici ; nous n'avons besoin que du décodage. (Quand les étiquettes ne sont pas observées, les HMM sont entraînés par EM, l'algorithme de Baum-Welch ; la première page de Bishop Sec. 13.2 introduit le modèle, et nous ne le dérivons pas ici.)

## De l'étiquetage morpho-syntaxique à la NER : les étiquettes BIO

Une entité peut s'étendre sur plusieurs mots, donc l'ensemble d'étiquettes encode les frontières : `B-X` commence une entité de type X, `I-X` la continue, `O` est en dehors de toute entité. « Maria Lopez visited New York » donne `B-PER I-PER O B-LOC I-LOC`. La NER est alors de l'étiquetage de séquences ordinaire avec $K=2\cdot\#\text{types}+1$ étiquettes. Après le décodage, une petite fonction transforme la séquence d'étiquettes en segments. Certaines séquences d'étiquettes sont illégales (`O I-PER`, ou `B-PER I-LOC`) ; un HMM simple peut apprendre que ces transitions sont rares (faible $A_{jk}$), et un `I-X` isolé est généralement lu comme le début d'une entité.

## L'idée des CRF

Un HMM est *génératif* : il modélise $P(x,t)$ et chaque mot ne dépend que de sa propre étiquette, donc tu ne peux pas facilement utiliser des caractéristiques comme « commence par une majuscule », « finit par -ing », « le mot précédent est Mr. » ou le mot à droite. Un **champ aléatoire conditionnel** (CRF linéaire) modélise directement $P(t\mid x)$ :

$$
P(t\mid x)=\frac{1}{Z(x)}\exp\Big(\sum_{i}\big[\mathbf{w}\cdot f(t_{i-1},t_i,x,i)\big]\Big),
$$

avec des fonctions de caractéristiques $f$ arbitraires et un normaliseur $Z(x)$ calculé par une récurrence forward. Le décodage est encore Viterbi, avec les scores $\mathbf w\cdot f$ à la place de $\log A+\log B$. Les étiqueteurs neuronaux conservent ce squelette : un BiLSTM ou un transformer produit des scores par mot, une couche CRF ajoute les scores de transition, et Viterbi décode. Beaucoup d'étiqueteurs transformers modernes abandonnent le CRF et prennent un arg max par mot, sacrifiant un peu de cohérence pour la simplicité.

Une référence utile à connaître : étiqueter chaque mot avec son étiquette d'entraînement la plus fréquente. Elle est forte parce que la plupart des mots sont non ambigus ; la difficulté réside dans les mots ambigus et inconnus, qui sont exactement là où le contexte aide.

## Exemple travaillé

Étiquettes $\{\mathrm{DET},\mathrm{NOUN},\mathrm{VERB}\}$, mots $\{\text{the},\text{a},\text{dog},\text{fish},\text{bark},\text{swim}\}$ et

$$
\pi=(0.6,\,0.3,\,0.1),\qquad A=\begin{pmatrix}0.05&0.90&0.05\\0.10&0.30&0.60\\0.50&0.40&0.10\end{pmatrix}.
$$

Émissions : DET dit *the* avec 0.7 et *a* avec 0.3 ; NOUN dit *dog* 0.4, *fish* 0.3, *bark* 0.3 ; VERB dit *fish* 0.3, *bark* 0.4, *swim* 0.3. Étiquette « the fish bark ». *fish* et *bark* sont tous deux ambigus.

- Pas 1 : seul DET peut dire *the* : $\delta_1=(0.42,\,0,\,0)$ puisque $0.6\times0.7=0.42$.
- Pas 2 (*fish*) : NOUN : $0.42\times0.9\times0.3=0.1134$ ; VERB : $0.42\times0.05\times0.3=0.0063$ ; DET ne peut pas dire *fish*.
- Pas 3 (*bark*) : NOUN : le meilleur prédécesseur est NOUN ($0.1134\times0.3=0.03402$ contre $0.0063\times0.4=0.00252$), donc $\delta=0.03402\times0.3=0.010206$. VERB : meilleur prédécesseur NOUN, $0.1134\times0.6=0.06804$ contre $0.0063\times0.1$, donc $\delta=0.06804\times0.4=0.027216$.
- Le maximum est VERB avec $0.027216$ ; la remontée donne DET, NOUN, VERB.

Remarque comment le dernier choix combine l'émission (VERB dit *bark* avec 0.4, NOUN avec 0.3) et la forte transition NOUN vers VERB (0.6) : ce qui compte, c'est le chemin entier, pas un facteur isolé. L'exercice vérifie Viterbi contre l'énumération par force brute des $3^n$ chemins sur ce modèle.

## Implémentation

```python
import numpy as np

def viterbi(obs, log_pi, log_A, log_B):
    n, K = len(obs), len(log_pi)
    delta = np.full((n, K), -np.inf)       # best log-prob ending in tag k at step i
    back = np.zeros((n, K), dtype=int)     # back-pointers
    delta[0] = log_pi + log_B[:, obs[0]]
    for i in range(1, n):
        cand = delta[i - 1][:, None] + log_A          # (K_prev, K_next)
        back[i] = cand.argmax(axis=0)
        delta[i] = cand.max(axis=0) + log_B[:, obs[i]]
    path = [int(delta[-1].argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return path[::-1], float(delta[-1].max())
```

## Formes

- `obs` : $(n,)$ identifiants entiers de mots. `log_pi` : $(K,)$. `log_A` : $(K,K)$, les lignes sont l'étiquette précédente. `log_B` : $(K,V)$.
- `delta`, `back` : $(n,K)$. `cand` : $(K,K)$, construit en diffusant une colonne $(K,1)$ contre $(K,K)$.

Bibliothèques : NLTK et spaCy fournissent des étiqueteurs pré-entraînés ; `sklearn-crfsuite` fournit des CRF ; étiqueteurs transformers via `token-classification` de Hugging Face. Évalue la NER avec la précision, le rappel et le F1 au niveau de l'entité (segment), pas l'exactitude par token : la plupart des tokens sont `O`, donc l'exactitude par token est flatteuse ([[nlp.02]] sur le déséquilibre).

## Idées reçues

- **« Viterbi choisit la meilleure étiquette pour chaque mot. »** Il choisit la meilleure *séquence* ; l'étiquette d'un mot peut différer de son étiquette individuellement la plus probable.
- **« Viterbi et l'algorithme forward sont les mêmes. »** Viterbi prend un max et donne un chemin ; forward prend une somme et donne $P(x)$.
- **« Les états du HMM sont les mots. »** Les états cachés sont les étiquettes ; les mots sont les observations.
- **« Une hypothèse de Markov d'ordre un convient pour le langage. »** C'est une forte simplification, adéquate pour l'étiquetage, ce qui explique pourquoi les modèles plus riches utilisent le contexte de toute la phrase.
- **« L'exactitude par token est une bonne métrique de NER. »** Utilise le F1 au niveau de l'entité.

## Points clés à retenir

- Un HMM factorise $P(x,t)$ en tables initiale, de transition et d'émission, apprenables par comptage plus lissage.
- Viterbi est de la programmation dynamique sur $\delta_i(k)$ avec pointeurs arrière, coût $O(nK^2)$, en espace logarithmique.
- Les étiquettes BIO transforment l'extraction d'entités en étiquetage de séquences ; décoder, puis convertir en segments.
- Les CRF modélisent $P(t\mid x)$ avec des caractéristiques riches ; les étiqueteurs modernes gardent la structure mais apprennent les scores.

## Explique-le toi-même

1. Pourquoi le meilleur chemin ne peut-il pas être trouvé en choisissant indépendamment la meilleure étiquette de chaque mot ? Utilise un mot ambigu dans ta réponse.
2. Pourquoi suffit-il de garder un chemin champion par étiquette et par position ?

## Pour aller plus loin

- Jurafsky et Martin, *Speech and Language Processing*, chapitres sur l'étiquetage morpho-syntaxique, les HMM et la NER.
- Bishop, *Pattern Recognition and Machine Learning*, Sec. 13.2 (forward-backward, Viterbi, EM pour les HMM).

## Critères de maîtrise

- Tu sais écrire les trois tables du HMM, leurs formes, et la probabilité jointe d'un étiquetage.
- Tu sais remplir à la main une table de Viterbi et remonter.
- Tu sais convertir des étiquettes BIO en segments d'entités et expliquer pourquoi le F1 au niveau du segment est utilisé.
