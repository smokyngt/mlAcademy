---
id: aud.04
title: "Reconnaissance automatique de la parole : CTC, RNN-T, attention et Whisper"
summary: "Résoudre le problème d'alignement entre de nombreuses trames audio et peu de caractères : dériver l'algorithme avant de la CTC et l'implémenter avec un décodage glouton et une recherche en faisceau sur préfixes, puis comparer RNN-T et les encodeurs-décodeurs à attention (Whisper) et tout mesurer avec le taux d'erreur sur les mots."
minutes: 110
teaches: [aud.ctc, aud.asr, aud.wer]
requires: [aud.mel, aud.classify, prob.basics, info.entropy, nn.layer, trf.architectures, alg.summation]
sources:
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/04-speech-recognition-asr/docs/en.md (three formulations CTC / RNN-T / attention, conditional independence of CTC, greedy collapse, beam-search skeleton, WER dynamic programme, deployment table, pitfalls, exercises). READ."
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/05-whisper-architecture-finetuning/docs/en.md (30 s log-mel window, 3000 frames, conv stride 2, encoder/decoder sizes, special-token prompt, timestamps tokens, chunking, hallucination pitfalls). READ."
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/17-audio-evaluation-metrics/docs/en.md (WER, CER, RTFx, first-token latency; normalisation before scoring). READ."
  - "math/coursAvanceApprentissageAudio.pdf, PDF p.15-19 (speech processing specifics, segmentation with CTC loss and greedy/beam decoding, transcription, Wav2Letter) and p.25-27 (autoregressive factorisation of waveform probability, WaveNet). READ."
---

## Où cela s'inscrit

Tu dis « allume les lumières de la cuisine » et un appareil transforme du texte en action. Derrière, il y a la reconnaissance automatique de la parole (ASR) : dix secondes d'audio, environ mille trames de caractéristiques issues de [[aud.02]], doivent devenir une courte chaîne d'une trentaine de caractères. La classification ([[aud.03]]) donnait une étiquette par extrait ; ici la sortie est une *séquence* dont la longueur est inconnue et dont les parties ne sont pas repérées dans le temps. Cette leçon dérive la fonction de perte qui a résolu ce problème (la CTC), présente ses deux successeurs (le transducteur RNN-T et les encodeurs-décodeurs à attention comme Whisper), et construit la métrique (le taux d'erreur sur les mots) servant à les comparer. Elle n'utilise les idées d'auto-attention et d'encodeur-décodeur des leçons sur le transformer (concepts `trf.attention` et `trf.architectures`) que dans la dernière partie.

## L'idée en termes simples

Imagine qu'on te donne une longue bande de film et une légende d'une ligne, et qu'on te demande de dire quelles images montrent quels mots. Personne ne l'a marqué. Le mot « okay » peut durer 200 ms ou 1200 ms ; il y a des silences, des hésitations et des respirations entre les mots. Si tu devais entraîner un réseau, un plan naïf serait de faire étiqueter chaque trame avec sa lettre par des humains, ce qui est extrêmement coûteux.

La **CTC** (Connectionist Temporal Classification) évite l'étiquetage. Le réseau regarde chaque trame et écrit une lettre ou une marque spéciale « rien ici » (le *blank*). En relisant la bande, tu fusionnes les lettres égales voisines et tu effaces les blanks. Beaucoup de gribouillis trame par trame différents se réduisent à la même légende, par exemple `h h _ e e l _ l o` et `_ h e _ l l _ l o o` se lisent tous deux « hello ». La CTC entraîne le réseau à maximiser la **probabilité totale de tous les gribouillis qui se réduisent à la bonne légende**, sans jamais en choisir un. Le blank n'est pas décoratif : sans lui, « hello » avec son double « l » ne pourrait pas s'écrire, car deux lettres égales à la suite fusionneraient.

Le **RNN-T** conserve cette idée mais ajoute un petit modèle de langage à l'intérieur : après avoir écrit une lettre, le réseau sait ce qu'il a déjà écrit et décide de ce qui vient ensuite. Les **encodeurs-décodeurs à attention** (Whisper) abandonnent complètement la vue trame par trame : un encodeur digère l'audio, et un décodeur écrit le texte morceau de mot par morceau de mot, en *regardant en arrière* à chaque fois tout l'audio avec l'attention.

## CTC : le modèle

Soit l'encodeur acoustique qui produit, pour chacune des $T$ trames, une distribution de probabilité sur $V+1$ symboles : $V$ vrais jetons (caractères ou morceaux de mots) plus le blank $\varnothing$ (indice 0 dans le code). Note $p_t(v)$ la probabilité du symbole $v$ à la trame $t$, lignes d'une matrice $\mathbf P\in\mathbb R^{T\times(V+1)}$ (softmax des sorties du réseau, donc chaque ligne somme à 1).

Un **alignement** (chemin) est une suite $\pi=(\pi_1,\dots,\pi_T)$ de symboles, un par trame. En supposant que les trames sont conditionnellement indépendantes sachant l'audio (l'hypothèse centrale de la CTC), sa probabilité est

$$
P(\pi\mid x)=\prod_{t=1}^{T}p_t(\pi_t).
$$

L'**application de réduction** $\mathcal B$ fusionne d'abord les symboles répétés consécutifs puis supprime les blanks : $\mathcal B(a\,a\,\varnothing\,a\,b\,b\,\varnothing\,c)=(a\,a\,b\,c)$. La probabilité d'une séquence d'étiquettes $y$ (la transcription) est la somme sur tous les chemins qui se réduisent à elle :

$$
P(y\mid x)=\sum_{\pi\in\mathcal B^{-1}(y)}\ \prod_{t=1}^{T}p_t(\pi_t),\qquad
\mathcal L_{\text{CTC}}=-\log P(y\mid x).
$$

Il y a un nombre exponentiel de chemins (jusqu'à $(V+1)^T$), donc nous avons besoin d'une récurrence.

## Dérivation : l'algorithme avant (forward)

**Séquence d'étiquettes étendue.** Insère un blank avant, entre et après les $U$ jetons cibles $y_1,\dots,y_U$ :

$$
l'=(\varnothing,\,y_1,\,\varnothing,\,y_2,\,\dots,\,\varnothing,\,y_U,\,\varnothing),\qquad |l'|=S=2U+1 .
$$

Numérote les positions $s=1,\dots,S$ ; les $s$ impairs sont des blanks, les $s$ pairs sont des étiquettes. Tout chemin valide parcourt $l'$ de gauche à droite : à chaque trame il reste à la position courante ou avance, et son symbole à la trame $t$ est $l'_s$ pour sa position courante $s$.

**Définition.** $\alpha_t(s)$ est la probabilité totale de tous les préfixes de chemins de longueur $t$ qui (i) se réduisent aux $\lfloor s/2\rfloor$ premiers jetons cibles et (ii) sont à la position $s$ (symbole $l'_s$) à la trame $t$.

**Initialisation.** Un chemin doit commencer dans le premier blank ou dans la première étiquette :

$$
\alpha_1(1)=p_1(\varnothing),\qquad \alpha_1(2)=p_1(y_1),\qquad \alpha_1(s)=0\ \ (s>2).
$$

**Récurrence.** Où un chemin à la position $s$ à la trame $t$ pouvait-il se trouver à la trame $t-1$ ?

1. En $s$ lui-même (le même symbole répété : les répétitions fusionnent).
2. En $s-1$ (avancer d'une position : d'un blank vers l'étiquette suivante, ou d'une étiquette vers le blank suivant).
3. En $s-2$ (sauter par-dessus un blank) : permis **seulement si** $l'_s$ est une étiquette (pas un blank) et $l'_s\ne l'_{s-2}$. Si les deux étiquettes étaient égales ($y_u=y_{u+1}$, par ex. le double « l »), sauter le blank les fusionnerait en une seule ; un blank entre elles est obligatoire. Sauter vers un blank par-dessus une étiquette effacerait cette étiquette.

Chaque cas multiplie par la probabilité du nouveau symbole à la trame $t$ :

$$
\alpha_t(s)=p_t(l'_s)\Big[\alpha_{t-1}(s)+\alpha_{t-1}(s-1)+\mathbb 1\{l'_s\neq\varnothing,\ l'_s\ne l'_{s-2}\}\;\alpha_{t-1}(s-2)\Big]
$$

(avec $\alpha_{t-1}(0)=\alpha_{t-1}(-1)=0$).

**Terminaison.** Un chemin complet se termine dans la dernière étiquette ou dans le blank final :

$$
P(y\mid x)=\alpha_T(S)+\alpha_T(S-1).
$$

Coût : $O(T\cdot U)$ au lieu de $O((V+1)^T)$. Si $T<U+r$ où $r$ est le nombre de paires adjacentes d'étiquettes cibles égales (chacune exige un blank entre elles), aucun chemin n'existe et la probabilité est exactement 0 (la perte est infinie). Par exemple, la cible $(1,1,1)$ exige au moins $T=5$ trames : `1 _ 1 _ 1`.

**Stabilité numérique.** Les produits débordent vers zéro (underflow) pour les longs énoncés, donc les implémentations travaillent en espace logarithmique : remplace les sommes par `logaddexp` et les produits par des sommes, $\log\alpha_t(s)=\log p_t(l'_s)+\operatorname{logaddexp}(\dots)$.

**Gradients.** Une récurrence arrière symétrique $\beta_t(s)$ donne la probabilité a posteriori que la trame $t$ soit à la position $s$, $\gamma_t(s)=\alpha_t(s)\beta_t(s)/P(y\mid x)$, et le gradient de la perte par rapport aux logits est la sortie du réseau moins la masse a posteriori de chaque symbole. Si tu implémentes la passe avant avec des opérations différentiables, un système d'autograd produit le même gradient. Le code de production (`torch.nn.CTCLoss`, warp-ctc) fusionne tout cela dans un noyau.

## Exemple détaillé 1 : trois trames, une lettre

Vocabulaire : le blank et une lettre $a$. Chaque trame a $p(\varnothing)=0.6$, $p(a)=0.4$. Cible $y=(a)$, donc $l'=(\varnothing,a,\varnothing)$, $S=3$, $T=3$.

| $t$ | $\alpha_t(1)$ ($\varnothing$) | $\alpha_t(2)$ ($a$) | $\alpha_t(3)$ ($\varnothing$) |
|---|---|---|---|
| 1 | 0.6 | 0.4 | 0 |
| 2 | 0.36 | 0.40 | 0.24 |
| 3 | 0.216 | 0.304 | 0.384 |

Vérifions une cellule : $\alpha_2(2)=p(a)[\alpha_1(2)+\alpha_1(1)]=0.4\,(0.4+0.6)=0.4$. Terminaison : $P(a)=0.384+0.304=0.688$.

Les trois transcriptions possibles : $P(\text{""})=0.6^3=0.216$ (tous blanks), $P(a)=0.688$, $P(aa)=0.4\cdot0.6\cdot0.4=0.096$ (seulement le chemin $a\,\varnothing\,a$). Leur somme vaut $1$.

## Décodage

**Glouton (meilleur chemin).** Prends le symbole le plus probable à chaque trame, puis réduis. Pour l'exemple ci-dessus, le meilleur symbole par trame est le blank, donc le décodage glouton renvoie la **chaîne vide**, alors que la transcription $a$ a une probabilité de $0.688$ contre $0.216$ pour la vide. Le meilleur *chemin* n'est pas la meilleure *transcription*, car la probabilité de la transcription est une somme sur de nombreux chemins. Le glouton est rapide et souvent suffisant quand le réseau est confiant.

**Recherche en faisceau sur préfixes.** Garde les $B$ meilleurs *préfixes* (transcriptions partielles), pas les chemins. Pour chaque préfixe, stocke deux log-probabilités : $p_b$ (alignements se terminant par un blank) et $p_{nb}$ (se terminant par une étiquette). À chaque trame, pour chaque préfixe $g$ de total $p=p_b+p_{nb}$ :

- blank : $g$ reste, ajoute $p\cdot p_t(\varnothing)$ à son $p_b$ ;
- étiquette $c$ égale à la dernière étiquette de $g$ : soit rester en $g$ (pas de blank entre, fusionné) en ajoutant $p_{nb}\,p_t(c)$ au $p_{nb}$ de $g$, soit étendre en $g{+}c$ (un blank les a séparées) en ajoutant $p_b\,p_t(c)$ au $p_{nb}$ du nouveau préfixe ;
- toute autre étiquette $c$ : étendre en $g{+}c$, en ajoutant $p\,p_t(c)$ à son $p_{nb}$.

Fusionner les alignements qui atteignent le même préfixe est ce que la recherche en faisceau « naïve » de la leçon du dépôt manque (son squelette ne fusionne pas, comme le reconnaît son propre exercice 2). Avec un faisceau assez large pour tout garder, la recherche en faisceau sur préfixes est **exacte** : notre test la compare à la force brute. Un modèle de langage externe peut être ajouté au score (« shallow fusion »).

Comme chaque trame est prédite indépendamment, la CTC n'a pas de modèle de langage interne : elle peut écrire un texte phonétiquement plausible mais agrammatical. C'est pourquoi les systèmes pratiques la combinent avec un modèle de langage externe ou choisissent le RNN-T ou l'attention.

## RNN-T : l'idée

Le transducteur a trois parties : un **encodeur** sur l'audio (trames $f_t$), un **réseau de prédiction** sur les étiquettes écrites jusqu'ici (état $g_u$ après $u$ étiquettes, en pratique un modèle de langage), et un **joiner** qui combine $f_t$ et $g_u$ en une distribution sur $V+1$ symboles. Imagine une grille avec le temps $t=0..T$ sur un axe et les étiquettes $u=0..U$ sur l'autre. À un nœud, le joiner soit émet l'étiquette $y_{u+1}$ (passer $u\to u+1$ sans consommer de trame), soit émet un blank (passer à la trame suivante). La probabilité d'une transcription est la somme sur tous les chemins monotones de $(0,0)$ à $(T,U)$ (leur nombre est $\binom{T+U}{U}$ en finissant par un blank), calculée par le même genre de récurrence avant que la CTC. Différences : (i) l'étiquette suivante dépend des étiquettes précédentes, ce qui corrige l'hypothèse d'indépendance de la CTC ; (ii) plusieurs étiquettes par trame sont possibles ; (iii) le treillis de la perte a $T\times(U+1)\times(V+1)$ entrées par énoncé au lieu de $T\times(V+1)$.

**Exemple détaillé 2 : pourquoi l'entraînement du RNN-T consomme beaucoup de mémoire.** Prenons $T=1500$ trames, $U=100$ jetons, $V+1=1000$ symboles, flottants de 4 octets : le treillis du transducteur contient $1500\cdot101\cdot1000=151.5\times10^6$ valeurs (environ 0,6 Go par énoncé, avant les gradients), la matrice de la CTC $1.5\times10^6$ valeurs (6 Mo). Ces nombres sont notre illustration ; les tailles réelles dépendent de la fréquence de trames de l'encodeur et du vocabulaire. Le RNN-T reste la norme pour la reconnaissance en flux continu sur appareil car il émet des jetons au fur et à mesure que l'audio arrive.

## Encodeurs-décodeurs à attention et Whisper

Un encodeur-décodeur à attention modélise la transcription de façon autorégressive,

$$
P(y\mid x)=\prod_{u=1}^{U}P\big(y_u\mid y_{<u},\ \mathrm{Enc}(x)\big),
$$

(la même factorisation par la règle de la chaîne que le cours en français écrit pour la synthèse de forme d'onde dans sa section WaveNet). L'encodeur transforme les trames log-mel en vecteurs cachés `(T', d)` ; le décodeur est un transformer causal qui, à chaque pas, fait de la **cross-attention** sur tous les vecteurs de l'encodeur (formes comme dans les leçons sur le transformer : requêtes du décodeur `(u, d)`, clés et valeurs de l'encodeur `(T', d)`). Aucun alignement n'est imposé : l'attention peut regarder n'importe où. L'entraînement est une simple entropie croisée avec forçage par l'enseignant (teacher forcing). Coûts : les jetons de sortie sont produits un par un, et le modèle a besoin de toute l'entrée, donc il n'est pas naturellement en flux continu.

**Whisper** (tel que décrit dans la leçon du dépôt) est un tel modèle :

- Entrée : 30 s d'audio sous forme de spectrogramme log-mel à 80 canaux avec un saut de 10 ms : $30/0.01=3000$ trames `(80, 3000)`. Les extraits plus courts sont complétés, les plus longs découpés en morceaux.
- Encodeur : un tronc convolutif de pas 2 divise la longueur par deux pour obtenir $3000/2=1500$ positions (une par 20 ms), puis des blocs transformer. Le coût de l'auto-attention par tête et par couche est de $1500^2=2.25\times10^6$ scores ; pour un encodeur de 32 couches et 20 têtes (la grande configuration citée dans la leçon du dépôt), cela fait $1.44\times10^9$ scores d'attention par fenêtre de 30 secondes.
- Décodeur : un décodeur transformer sur un vocabulaire d'environ 51 865 jetons de type byte-pair.
- **Multitâche par prompt :** des jetons spéciaux tels que `<|startoftranscript|>`, une balise de langue, `<|transcribe|>` ou `<|translate|>`, et `<|notimestamps|>` orientent un seul modèle vers plusieurs tâches ; les jetons d'horodatage marquent des décalages de 20 ms. Changer le jeton de langue change la langue.
- Les statistiques log-mel doivent être celles de Whisper (pas les valeurs par défaut de librosa), sinon la sortie se dégrade.

Modes de défaillance connus dans la leçon du dépôt : du texte halluciné (« thanks for watching ») sur du silence, c'est pourquoi un détecteur d'activité vocale filtre l'audio ; une répétition incontrôlée quand chaque fenêtre de 30 s est conditionnée par le texte précédent ; et les extraits courts complétés. Les chiffres de benchmark rapportés pour 2026 (environ 1,4 à 1,8 % de WER sur LibriSpeech test-clean pour les meilleurs systèmes) proviennent de la leçon du dépôt et n'ont pas été vérifiés indépendamment ici ; le classement entre les meilleurs systèmes tient à quelques dixièmes de pourcent, et les propriétés de déploiement (flux continu, latence, couverture des langues) comptent davantage.

| Formulation | Alignement | LM interne | Flux continu | Usage typique |
|---|---|---|---|---|
| CTC | marginalisé sur les chemins | non | oui | modèles rapides, simples, de type wav2vec |
| RNN-T | marginalisé sur le treillis | oui (réseau de prédiction) | oui | sur appareil, faible latence |
| Enc-déc à attention | appris par l'attention | oui (décodeur) | pas nativement | qualité hors ligne (Whisper) |

## Taux d'erreur sur les mots et distance d'édition

Compare une référence $r_1\dots r_N$ avec une hypothèse $h_1\dots h_M$. La **distance d'édition** (Levenshtein) est le nombre minimal de substitutions (S), de suppressions (D) et d'insertions (I) pour transformer l'une en l'autre. Définis $D[i][j]$ comme la distance entre les $i$ premiers mots de la référence et les $j$ premiers mots de l'hypothèse :

$$
D[i][0]=i,\qquad D[0][j]=j,
$$

$$
D[i][j]=\min\Big(D[i-1][j]+1,\ D[i][j-1]+1,\ D[i-1][j-1]+\mathbb 1\{r_i\ne h_j\}\Big).
$$

Les trois termes sont : supprimer $r_i$ ; insérer $h_j$ ; correspondre ou substituer. Alors

$$
\text{WER}=\frac{S+D+I}{N}=\frac{D[N][M]}{N}.
$$

Remonter dans la table depuis $(N,M)$ restitue un alignement et les décomptes $S,D,I$ (des alignements de coût égal peuvent répartir le total différemment ; le total, lui, est unique). Le WER n'est **pas borné par 1** : une hypothèse avec beaucoup d'insertions peut dépasser 100 %. Le **taux d'erreur sur les caractères** (CER) applique la même formule aux caractères et est préféré pour les langues sans frontières de mots. Normalise toujours avant de noter (minuscules, suppression de la ponctuation, unification des nombres), sinon les différences de format comptent comme des erreurs.

**Exemple détaillé 3.** Référence `the cat sat on mat`, hypothèse `the cat on a mat`. La table (lignes : préfixes de la référence, colonnes : préfixes de l'hypothèse, en commençant par le préfixe vide) se termine par $D[5][5]=2$ :

$$
\begin{matrix}
 & \varnothing & \text{the} & \text{cat} & \text{on} & \text{a} & \text{mat}\\
\varnothing & 0&1&2&3&4&5\\
\text{the}&1&0&1&2&3&4\\
\text{cat}&2&1&0&1&2&3\\
\text{sat}&3&2&1&1&2&3\\
\text{on}&4&3&2&1&2&3\\
\text{mat}&5&4&3&2&2&2
\end{matrix}
$$

Donc WER $=2/5=0.4$ (supprimer « sat », insérer « a »). Un second exemple : référence `please turn on the kitchen lights`, hypothèse `please turn the kitchen light on now` donne une distance de 4, donc WER $=4/6=0.667$ (une suppression, une substitution, deux insertions). Note une correction à la leçon du dépôt : elle annote le score de `Please turn on the lights.` contre `please turn on the light` comme « ~0.17 », mais après normalisation il y a 5 mots de référence et 1 erreur, soit **0.2**.

Le WER pondère tous les mots à égalité (un « not » omis compte comme un « the » omis) et ne dit rien du sens ; en production, suis aussi la latence (RTFx, latence du premier jeton) et l'erreur par domaine.

## Implémentation

```python
import numpy as np
from collections import defaultdict

NEG = -np.inf

def collapse(path, blank=0):
    out, prev = [], None
    for p in path:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out

def ctc_forward(logp, target, blank=0):
    """logp: (T, V) log-probabilities (rows normalised). Returns log P(target | x)."""
    T = logp.shape[0]
    ext = [blank]
    for c in target:
        ext += [c, blank]                       # length S = 2U + 1
    S = len(ext)
    alpha = np.full((T, S), NEG)
    alpha[0, 0] = logp[0, blank]
    if S > 1:
        alpha[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            terms = [alpha[t - 1, s]]
            if s >= 1:
                terms.append(alpha[t - 1, s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                terms.append(alpha[t - 1, s - 2])
            alpha[t, s] = np.logaddexp.reduce(terms) + logp[t, ext[s]]
    end = alpha[T - 1, S - 1]
    if S > 1:
        end = np.logaddexp(end, alpha[T - 1, S - 2])
    return end

def ctc_greedy(logp, blank=0):
    return collapse(logp.argmax(axis=1), blank)

def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h)[len(r), len(h)] / max(1, len(r))
```

Les exercices ajoutent `ctc_prefix_beam_search`, la remontée S/D/I et des tests contre l'énumération par force brute.

### Formes

| Objet | Forme |
|---|---|
| sortie de l'encodeur / `logp` | `(T, V+1)` (CTC : log-softmax sur les symboles) |
| étiquettes étendues $l'$ et $\alpha$ | $S=2U+1$ ; `alpha` est `(T, S)` |
| treillis du transducteur | `(T, U+1, V+1)` |
| entrée de Whisper / sortie de l'encodeur | `(80, 3000)` / `(1500, d)` |
| table de distance d'édition | `(N+1, M+1)` |

**Ce que font les bibliothèques.** `torch.nn.CTCLoss` et `torchaudio.functional.rnnt_loss` implémentent les pertes avec des noyaux fusionnés ; `pyctcdecode`/`torchaudio` proposent la recherche en faisceau sur préfixes avec fusion KenLM ; `jiwer` calcule le WER avec des transformations de normalisation ; `openai-whisper`, `faster-whisper` et WhisperX fournissent l'inférence, l'accélération à l'exécution et la gestion des longs enregistrements (filtrage VAD, alignement des mots).

## Idées reçues courantes

- **« La CTC choisit le meilleur alignement. »** Elle somme sur tous les alignements pour l'entraînement ; le décodeur glouton choisit le meilleur chemin unique, qui peut différer de la meilleure transcription (exemple détaillé 1).
- **« Le blank signifie silence. »** Il signifie « aucune nouvelle étiquette émise à cette trame » ; les réseaux mettent souvent des blanks sur la plupart des trames, même pendant la parole.
- **« La sortie de la CTC conserve la longueur. »** Le chemin a $T$ symboles ; la transcription est plus courte après réduction.
- **« Un WER plus bas signifie de meilleures transcriptions pour tout usage. »** Le WER ignore le sens, la ponctuation et le locuteur ; compare sur ton domaine, avec une normalisation identique.
- **« Whisper entend nativement des fenêtres de 30 secondes, donc le flux continu est facile. »** La fenêtre est fixe ; le flux continu exige une ingénierie supplémentaire ou une autre architecture.
- **« Recherche en faisceau sur préfixes = recherche en faisceau ordinaire sur les trames. »** Sans fusion des chemins qui se réduisent au même préfixe, le faisceau se remplit de doublons.

## Points clés à retenir

- La CTC marginalise sur les alignements augmentés de blanks : $P(y|x)=\sum_{\pi\in\mathcal B^{-1}(y)}\prod_tp_t(\pi_t)$, calculée par la récurrence avant en $O(TU)$ sur $l'$ avec la règle de saut $l'_s\ne\varnothing,\ l'_s\ne l'_{s-2}$.
- Meilleur chemin $\ne$ meilleure transcription : la recherche en faisceau sur préfixes (exacte avec un faisceau large) suit les probabilités blank et non-blank par préfixe.
- Le RNN-T ajoute un réseau d'historique des étiquettes (un LM intégré) au prix d'un treillis 3D ; les encodeurs-décodeurs à attention abandonnent les contraintes d'alignement et décodent de façon autorégressive (Whisper : 30 s, 3000 trames, 1500 positions d'encodeur, jetons de tâche).
- WER $=D[N][M]/N$ à partir de la récurrence de Levenshtein ; il peut dépasser 1 ; normalise d'abord.

## Explique-le toi-même

1. Pourquoi faut-il insérer un blank entre deux lettres égales dans la séquence étendue, et comment cela apparaît-il dans la règle de saut de la récurrence avant ?
2. Décris une situation où le décodage CTC glouton et la transcription la plus probable diffèrent, et dis laquelle la recherche en faisceau sur préfixes trouve.

## Pour aller plus loin

- Dépôt, phase 06, leçons 04, 05 et 17 ; les leçons citent Graves et al. (2006) pour la CTC, Graves (2012) pour le RNN-T et Radford et al. (2022) pour Whisper (non lus directement).
- [[aud.05]] fait fonctionner le pipeline dans l'autre sens : du texte à la parole.
- Hannun, « Sequence Modeling with CTC » (Distill, 2017) pour des images du treillis ; nous ne l'avons pas lu, et la description de la recherche sur préfixes ci-dessus suit notre propre dérivation que l'exercice teste contre la force brute.

## Critères de maîtrise

Tu sais (a) énoncer l'application de réduction et la perte de la CTC, (b) dériver la récurrence avant avec la règle de saut et la terminaison, (c) l'implémenter en espace logarithmique et la vérifier contre l'énumération par force brute, (d) expliquer pourquoi le décodage glouton peut être sous-optimal et implémenter la recherche en faisceau sur préfixes, (e) comparer la CTC, le RNN-T et les encodeurs-décodeurs à attention, et (f) calculer le WER et le CER avec une table de distance d'édition.
