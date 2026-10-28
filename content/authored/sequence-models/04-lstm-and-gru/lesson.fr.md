---
id: seq.04
title: LSTM et GRU
summary: "Pourquoi un chemin additif d'état de cellule répare le gradient évanescent, toutes les équations de portes avec leurs dimensions pour LSTM et GRU, les nombres de paramètres, un pas de LSTM calculé à la main avec sa passe arrière, et des implémentations NumPy vérifiées par différences finies."
minutes: 100
teaches: [seq.lstm, seq.gru]
requires: [seq.rnn, seq.rnn-gradients, nn.activation, nn.backprop]
sources:
  - "math/cours-tal-02.pdf (CNAM RCP217, PDF p.21-25: the LSTM idea of a second memory path, LSTM equations f/i/o/c/h, the LSTM and vanishing gradient slide with d c_t / d c_{t-1}, GRU as a simpler alternative, stacked layers). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/docs/en.md (LSTM and GRU described as gated cells, the cell-state highway, bidirectional RNNs, max-pooling over states). READ."
---

## Objectifs d'apprentissage

- Expliquer, avec une dérivée, pourquoi un état de cellule additif à portes laisse les gradients voyager loin.
- Écrire toutes les équations de portes du LSTM ($i,f,g,o$) et du GRU ($r,z$) avec leurs dimensions.
- Compter les paramètres des couches RNN, LSTM et GRU et les comparer.
- Rétropropager à travers un pas de LSTM à la main et en code.
- Vérifier des propriétés des portes (un grand biais d'oubli conserve la cellule, une porte de mise à jour à 1 conserve l'état) et vérifier un pas par gradient numérique.

## Prérequis

La récurrence du RNN, la BPTT et l'argument du produit de jacobiennes de [[seq.03]] ; les fonctions d'activation et les dérivées de la sigmoïde/tanh.

## Où cela s'inscrit

Les réseaux récurrents ont été le cheval de trait de la modélisation du langage, de la parole et des séries temporelles d'environ 2014 à 2017 (traduction, reconnaissance vocale, écriture manuscrite), et le LSTM en est la raison : il a rendu possible l'entraînement sur des séquences de centaines de pas. L'idée se transmet aux architectures modernes, où l'« autoroute » additive réapparaît sous forme de connexion résiduelle ([[seq.01]]) dans chaque bloc de transformer. Comprendre pourquoi le LSTM fonctionne est l'une des meilleures façons de comprendre pourquoi les chemins résiduels fonctionnent.

## L'idée en mots simples

Retour au pense-bête de [[seq.03]]. Dans un RNN simple, tu réécris *toute* la note à chaque mot. Après 50 mots, la première chose que tu as écrite a été écrasée et brouillée 50 fois.

Un LSTM te donne **un carnet plus un secrétariat de portes**. Le carnet ($c_t$, l'état de cellule) est transmis inchangé à moins que quelque chose ne lui dise le contraire. Trois petites décisions, chacune un nombre entre 0 et 1 par emplacement mémoire, le contrôlent :

- **Porte d'oubli** : quelle part de chaque ancienne entrée je garde ? (1 : tout garder, 0 : effacer.)
- **Porte d'entrée** : quelle part du nouveau contenu candidat j'écris ?
- **Porte de sortie** : quelle part du carnet je montre au reste du réseau maintenant ?

L'essentiel est que la mise à jour est **additive** : nouveau carnet = (ancien carnet conservé) + (contenu nouvellement écrit). Comme c'est une somme, et non un écrasement répété, l'information comme le gradient peuvent voyager le long du carnet pendant longtemps quand la porte d'oubli reste proche de 1.

Le **GRU** est une version plus légère : pas de carnet séparé, un seul état, et deux portes. Une *porte de mise à jour* décide quelle part de l'ancien état garder plutôt que remplacer ; une *porte de réinitialisation* décide quelle part de l'ancien état le nouveau contenu proposé peut regarder.

## Définitions formelles

Entrée $x_t\in\mathbb{R}^n$, taille cachée $H$, précédents $h_{t-1},c_{t-1}\in\mathbb{R}^H$. $\sigma$ est la sigmoïde logistique et $\odot$ le produit terme à terme.

### LSTM

$$
\begin{aligned}
&i_t=\sigma(W_ix_t+U_ih_{t-1}+b_i) &&\text{input gate}\\
&f_t=\sigma(W_fx_t+U_fh_{t-1}+b_f) &&\text{forget gate}\\
&g_t=\tanh(W_gx_t+U_gh_{t-1}+b_g) &&\text{candidate}\\
&o_t=\sigma(W_ox_t+U_oh_{t-1}+b_o) &&\text{output gate}
\end{aligned}
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t).
$$

(Les diapositives du CNAM appellent le candidat $\tanh(W_cx_t+U_ch_{t-1}+b_c)$ et notent que $f_t$ oublie sélectivement tandis que $i_t$ laisse entrer de la nouvelle information.) Dimensions : chaque $W_\bullet:(H,n)$, chaque $U_\bullet:(H,H)$, chaque $b_\bullet:(H)$, toutes les portes et tous les états $(H)$. En code, les quatre blocs sont empilés : $W:(4H,n)$, $U:(4H,H)$, $b:(4H)$, un seul produit matriciel donne $a=Wx+Uh_{t-1}+b\in\mathbb{R}^{4H}$ qui est découpé en quatre. Nous utilisons l'ordre $i,f,g,o$ (PyTorch utilise le même ordre).

### GRU

$$
\begin{aligned}
&r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r) &&\text{reset gate}\\
&z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z) &&\text{update gate}\\
&\tilde h_t=\tanh\big(W_nx_t+r_t\odot(U_nh_{t-1})+b_n\big) &&\text{candidate}\\
&h_t=(1-z_t)\odot\tilde h_t+z_t\odot h_{t-1}
\end{aligned}
$$

Empilé : $W:(3H,n)$, $U:(3H,H)$, $b:(3H)$ dans l'ordre $r,z,n$. Les conventions diffèrent d'un article à l'autre (quel côté reçoit $z$ ou $1-z$, si $r$ multiplie $h_{t-1}$ avant ou après le produit matriciel). La version ci-dessus est celle utilisée par le `GRUCell` de PyTorch (à son second vecteur de biais près) ; la vérification numérique contre PyTorch ci-dessous le confirme.

### Nombres de paramètres

Chaque bloc de porte (ou de candidat) a $Hn+H^2+H$ paramètres.

| Cellule | Blocs | Formule | $n=128$, $H=256$ |
|---|---|---|---|
| RNN | 1 | $H(n+H+1)$ | 98,560 |
| GRU | 3 | $3H(n+H+1)$ | 295,680 |
| LSTM | 4 | $4H(n+H+1)$ | 394,240 |

Un GRU a donc les 3/4 des paramètres d'un LSTM de même taille, et le LSTM en a 4 fois ceux du RNN simple. (PyTorch garde deux vecteurs de biais par bloc, donc son `LSTMCell(128,256)` en rapporte 395,264 et `GRUCell` 296,448 : $+4H$ et $+3H$.) Les diapositives du CNAM ajoutent que les GRU sont plus rapides et s'entraînent plus vite, « peut-être moins efficaces dans certains cas (le jury semble encore délibérer) ».

## Dérivation : pourquoi les portes aident les gradients

Dans un RNN simple, le chemin de $h_{t-1}$ à $h_t$ passe par une matrice et une fonction d'écrasement : $\partial h_t/\partial h_{t-1}=\operatorname{diag}(1-h_t^2)W_{hh}$, et le produit sur $T$ pas se comporte comme $\rho(W_{hh})^T$ ([[seq.03]]). Chaque pas *doit* passer par $W_{hh}$ et la non-linéarité.

Regarde plutôt le chemin de cellule du LSTM. $c_t=f_t\odot c_{t-1}+i_t\odot g_t$. Les portes et le candidat ne dépendent de $c_{t-1}$ qu'indirectement par $h_{t-1}=o_{t-1}\odot\tanh(c_{t-1})$. Séparons la dérivée en parties **directe** et **indirecte** (c'est la structure en somme montrée dans les diapositives, p. 23) :

$$
\frac{\partial c_t}{\partial c_{t-1}}=\underbrace{\operatorname{diag}(f_t)}_{\text{direct, elementwise}}+\underbrace{\operatorname{diag}(c_{t-1})\frac{\partial f_t}{\partial c_{t-1}}+\operatorname{diag}(i_t)\frac{\partial g_t}{\partial c_{t-1}}+\operatorname{diag}(g_t)\frac{\partial i_t}{\partial c_{t-1}}}_{\text{through }h_{t-1}} .
$$

Le terme direct ne contient **ni matrice de poids ni dérivée d'écrasement** : c'est simplement la porte d'oubli. Si l'on ne suit que ce terme sur $T$ pas, on obtient

$$
\frac{\partial c_T}{\partial c_0}\;\approx\;\prod_{t=1}^{T}\operatorname{diag}(f_t)\quad\Longrightarrow\quad \text{component } j:\ \prod_tf_{t,j}.
$$

Chaque composante est un produit de nombres que le réseau contrôle lui-même et peut pousser près de 1. Compare :

| facteur par pas | après $T=100$ pas |
|---|---|
| $0.9$ | $2.7\times10^{-5}$ |
| $0.99$ | $0.37$ |
| $\sigma(5)=0.9933$ | $0.51$ |

Si $f_t\approx1$ et que la porte d'entrée est $\approx0$, la cellule est copiée inchangée et le gradient la traverse inchangé. Le gradient qui atteint $c_0$ n'est pas forcé de passer par une fonction d'écrasement et une matrice répétée. C'est exactement le rôle du **biais de la porte d'oubli** : initialiser $b_f$ à une valeur positive (comme 1 à 5) démarre l'entraînement avec $f\approx$ grand, donc la mémoire est conservée par défaut. (Les termes indirects existent toujours et peuvent compter ; c'est un argument pour un chemin favorable, pas une garantie. Et remarque que $\sigma(1)^{100}\approx2\times10^{-14}$ : un biais de 1 seul ne préserve pas l'information sur 100 pas à moins que l'entraînement pousse $f$ plus haut.)

Même argument pour le GRU. À partir de $h_t=(1-z_t)\odot\tilde h_t+z_t\odot h_{t-1}$ :

$$
\frac{\partial h_t}{\partial h_{t-1}}=\operatorname{diag}(z_t)+(\text{terms through the gates and }\tilde h_t).
$$

Une porte de mise à jour proche de 1 copie l'état précédent. C'est la même idée d'identité additive que le bloc résiduel $y=x+F(x)$ de [[seq.01]], avec un coefficient apprenable et dépendant de l'entrée.

## Dérivation : rétropropagation à travers un pas de LSTM

Étant donnés $\bar h_t$ et $\bar c_t$ (gradients venant du futur et de la sortie au temps $t$), calcule tout avec la règle de la chaîne. Soit $\tau=\tanh(c_t)$.

1. À travers $h_t=o_t\odot\tau$ : $\ \bar o=\bar h\odot\tau$ et la part de $\bar c$ venant de $h_t$ est $\bar h\odot o\odot(1-\tau^2)$. Gradient total sur la cellule :

$$
\bar c^{\,tot}=\bar c_t+\bar h_t\odot o_t\odot(1-\tau^2).
$$

2. À travers $c_t=f\odot c_{t-1}+i\odot g$ :

$$
\bar f=\bar c^{\,tot}\odot c_{t-1},\quad \bar i=\bar c^{\,tot}\odot g,\quad \bar g=\bar c^{\,tot}\odot i,\quad \bar c_{t-1}=\bar c^{\,tot}\odot f .
$$

   Le dernier est le chemin additif : le gradient vers la cellule précédente est le gradient entrant multiplié par $f$, sans rien d'autre sur le chemin.

3. À travers les non-linéarités des portes (dérivée de la sigmoïde $s(1-s)$, dérivée de tanh $1-g^2$) :

$$
\bar a_i=\bar i\odot i\odot(1-i),\ \bar a_f=\bar f\odot f\odot(1-f),\ \bar a_g=\bar g\odot(1-g^2),\ \bar a_o=\bar o\odot o\odot(1-o).
$$

4. Empile $\bar a=[\bar a_i;\bar a_f;\bar a_g;\bar a_o]\in\mathbb{R}^{4H}$. Puisque $a=Wx+Uh_{t-1}+b$ :

$$
\bar W=\bar a\,x^\top\ (4H\times n),\quad \bar U=\bar a\,h_{t-1}^\top\ (4H\times H),\quad \bar b=\bar a,\quad \bar x=W^\top\bar a,\quad \bar h_{t-1}=U^\top\bar a .
$$

   À travers les pas de temps, $\bar W,\bar U,\bar b$ s'accumulent (poids partagés, comme dans [[seq.03]]), et $\bar h_{t-1}$, $\bar c_{t-1}$ sont les deux quantités transmises vers l'arrière.

## Exemple détaillé (un pas de LSTM, $n=H=1$)

$W=(0.5,\,1.0,\,0.8,\,0.3)^\top$ pour $(i,f,g,o)$, $U=(0.1,\,0.2,\,0.4,\,0.5)^\top$, $b=(0,\,1,\,0,\,0)^\top$ ; $x=1$, $h_{t-1}=0.5$, $c_{t-1}=1$.

- Pré-activations $a=Wx+Uh_{t-1}+b=(0.55,\ 2.10,\ 1.00,\ 0.55)$.
- Portes : $i=\sigma(0.55)=0.6341$, $f=\sigma(2.1)=0.8909$, $g=\tanh(1)=0.7616$, $o=\sigma(0.55)=0.6341$.
- Cellule : $c_t=0.8909\cdot1+0.6341\cdot0.7616=1.3739$ ; $\tanh(c_t)=0.8796$ ; $h_t=0.6341\cdot0.8796=0.5578$.

Passe arrière avec $\bar h_t=1$, $\bar c_t=0$ :

- $\bar c^{\,tot}=1\cdot0.6341\cdot(1-0.8796^2)=0.14354$
- $\bar c_{t-1}=\bar c^{\,tot}f=0.14354\cdot0.8909=0.12788$
- $\bar o=\tau=0.8796$, donc $\bar a_o=0.8796\cdot0.6341\cdot0.3659=0.2041$
- $\bar i=\bar c^{\,tot}g=0.1093$, donc $\bar a_i=0.1093\cdot0.6341\cdot0.3659=0.02536$

Le code de l'exercice renvoie $\bar c_{t-1}=0.12788$, $\bar b=(0.02536,\,0.01395,\,0.03823,\,0.20407)$, $\bar x=0.11844$, $\bar h_{t-1}=0.12265$.

## Implémentation

```python
import numpy as np

def sigmoid(s):
    return 0.5 * (1.0 + np.tanh(0.5 * s))

def lstm_step(x, h_prev, c_prev, W, U, b):          # W:(4H,n) U:(4H,H) b:(4H)
    H = h_prev.shape[0]
    a = W @ x + U @ h_prev + b
    i, f = sigmoid(a[:H]), sigmoid(a[H:2*H])
    g, o = np.tanh(a[2*H:3*H]), sigmoid(a[3*H:])
    c = f * c_prev + i * g
    tc = np.tanh(c)
    return o * tc, c, (x, h_prev, c_prev, W, U, i, f, g, o, tc)

def gru_step(x, h_prev, W, U, b):                   # W:(3H,n) U:(3H,H) b:(3H)
    H = h_prev.shape[0]
    r = sigmoid(W[:H] @ x + U[:H] @ h_prev + b[:H])
    z = sigmoid(W[H:2*H] @ x + U[H:2*H] @ h_prev + b[H:2*H])
    n_ = np.tanh(W[2*H:] @ x + r * (U[2*H:] @ h_prev) + b[2*H:])
    return (1 - z) * n_ + z * h_prev
```

La passe arrière du premier exercice, ce sont les cinq lignes de la dérivation ci-dessus.

### Vérification

- **Différences finies** : avec la perte scalaire $A^\top h_t+B^\top c_t$ pour des $A,B$ aléatoires, chaque gradient ($\bar x,\bar h_{t-1},\bar c_{t-1},\bar W,\bar U,\bar b$) concorde avec les différences centrées à environ $10^{-7}$ près dans les tests de l'exercice.
- **Contre PyTorch** (une vérification de brouillon, qui ne fait pas partie des exercices) : copier $W,U,b$ dans `torch.nn.LSTMCell` (second biais mis à 0) reproduit $h_t,c_t$ et tous les gradients à $\sim10^{-8}$ près (arrondi des poids en float32), et `torch.nn.GRUCell` reproduit notre $h_t$ à la précision machine. Cela confirme l'ordre des portes ($i,f,g,o$ et $r,z,n$) et que la convention de `GRUCell` est $h_t=(1-z)\odot\tilde h+z\odot h_{t-1}$.
- **Tests de comportement** (second et premier exercice) : avec des poids nuls et $b_f=+20$, $b_i=-20$, la cellule est inchangée et $\bar c_{t-1}=\bar c_t$ ; avec $b_f=b_i=-20$, la cellule est effacée ; pour le GRU, $b_z=+30$ donne $h_t=h_{t-1}$ et $b_z=-30$ donne $h_t=\tilde h$, et avec $b_r=b_z=-30$ la sortie ne dépend plus du tout de $h_{t-1}$.

## Idées reçues

- **« Les LSTM résolvent complètement les gradients évanescents. »** Ils les rendent bien plus doux grâce au chemin additif ; les très longues séquences (milliers de pas) restent difficiles, et les gradients explosifs demandent encore l'écrêtage.
- **« L'état de cellule est la sortie. »** La sortie est $h_t=o_t\odot\tanh(c_t)$ ; la cellule est une mémoire interne.
- **« Plus de portes signifie plus de puissance. »** Les GRU avec 3/4 des paramètres donnent souvent des performances similaires ; lequel est meilleur dépend de la tâche.
- **« Les portes sont décidées par une règle. »** Ce sont des fonctions apprises de $x_t$ et $h_{t-1}$, entraînées par rétropropagation comme tout le reste.
- **« Les portes suppriment le goulot d'étranglement séquentiel. »** Le calcul reste séquentiel en $t$ ; seul le chemin du gradient est amélioré.

## Limites et compromis

- Coût : 4 fois (LSTM) ou 3 fois (GRU) les paramètres et le calcul par pas d'un RNN simple avec le même $H$.
- La mémoire pour la BPTT stocke toutes les portes et tous les états à chaque pas : environ $7H$ valeurs par pas et par échantillon ($i,f,g,o,c,h,\tanh c$).
- Toujours strictement séquentiel, donc plus lent à entraîner que les convolutions ou l'attention sur le matériel moderne.
- Un seul état de taille fixe (et $c$) doit résumer le passé : pour la traduction cela devient un goulot d'étranglement ([[seq.05]]).

## Points clés

- LSTM : $c_t=f_t\odot c_{t-1}+i_t\odot g_t$, $h_t=o_t\odot\tanh(c_t)$ avec quatre blocs $i,f,g,o$ de taille $H$ ; empilé $W:(4H,n)$, $U:(4H,H)$.
- La dérivée du chemin de cellule contient le terme direct $\operatorname{diag}(f_t)$ sans matrice de poids ni écrasement : les gradients peuvent survivre quand $f\approx1$ (donc initialise le biais d'oubli positif).
- GRU : $h_t=(1-z_t)\odot\tilde h_t+z_t\odot h_{t-1}$, porte de réinitialisation $r_t$ dans le candidat ; 3 blocs, 3/4 des paramètres d'un LSTM.
- Passe arrière du LSTM : $\bar c^{tot}=\bar c+\bar h\odot o\odot(1-\tanh^2c)$, $\bar c_{t-1}=\bar c^{tot}\odot f$, puis les non-linéarités des portes et une matrice empilée en arrière.

## Explique-le toi-même

1. Pourquoi $\bar c_{t-1}=\bar c^{tot}\odot f_t$ donne-t-il un chemin de gradient qu'un RNN simple n'a pas ?
2. Quel paramètre initialiserais-tu à une grande valeur positive pour qu'un LSTM mémorise par défaut, et que t'attends-tu à ce que la cellule fasse lors des premiers pas d'entraînement ?

## Pour aller plus loin

- Diapositives CNAM RCP217 `cours-tal-02.pdf`, p. 21-25 du PDF.
- Bibliothèque source : `ai-engineering-from-scratch` phase 05 leçon 08.
- Ensuite : [[seq.05]], où un encodeur et un décodeur faits de telles cellules rencontrent le goulot d'étranglement qui a motivé l'attention.

## Critères de maîtrise

Tu sais écrire de mémoire les équations du LSTM et du GRU avec toutes les dimensions, dériver les cinq formules arrière d'un pas de LSTM, calculer les nombres de paramètres, et tes cellules NumPy passent une vérification de gradient et les tests de propriétés des portes.
