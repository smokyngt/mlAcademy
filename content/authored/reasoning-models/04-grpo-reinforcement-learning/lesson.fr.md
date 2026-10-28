---
id: rsn.04
title: "Apprentissage par renforcement pour le raisonnement : GRPO"
summary: Entraîner un modèle à raisonner en récompensant les réponses vérifiées comme correctes, dériver l'avantage relatif au groupe qui remplace un critique appris, assembler la perte GRPO avec clipping et régularisation KL, et vérifier numériquement son gradient sur une politique jouet.
minutes: 100
teaches: [rsn.grpo, rsn.reward-design]
requires: [rl.policy-gradient, rsn.verifier]
sources:
  - "reasoning-from-scratch/ch06/01_main-chapter-code/ch06_main.ipynb (Sec. 6.1-6.13): RL for LLMs, RLHF versus RLVR, GRPO procedure, rollouts, reward_rlvr, advantages, sequence log-probabilities (sum versus mean), the policy-gradient loss, compute_grpo_loss, the training loop, results table. READ."
  - "reasoning-from-scratch/ch07/01_main-chapter-code/ch07_main.ipynb (Sec. 7.2-7.6: metrics to track, clipped policy ratio, KL term, format reward and conditional format reward): READ."
  - "reasoning-from-scratch/ch03/01_main-chapter-code/ch03_main.ipynb (Sec. 3.4-3.7: the verifier that defines the reward): READ."
  - "reasoning-from-scratch/chE/chE_main.ipynb (Sec. E.6, E.8: batched GRPO rollouts and timings): READ."
---

## Où cela s'inscrit

Jusqu'ici le modèle était figé et nous dépensions du calcul au moment de la réponse ([[rsn.02]], [[rsn.03]]). Maintenant, nous changeons le modèle lui-même : **l'entraîner, par apprentissage par renforcement (RL), à produire des chaînes de pensée qui aboutissent à des réponses vérifiées comme correctes.** C'est la technique derrière les modèles de raisonnement récents (le livre suit la recette décrite publiquement pour DeepSeek-R1). Elle demande trois ingrédients que tu as déjà : une politique qui génère du texte ([[llm.02]]), un vérificateur qui la note ([[rsn.01]]) et l'idée du gradient de politique de [[rl.05]]. La nouveauté est la façon d'estimer à moindre coût « ce rollout était-il meilleur que d'habitude ? », sans entraîner un second réseau de neurones. C'est la **Group Relative Policy Optimisation (GRPO)**.

Deux leçons suivent celle-ci : [[rsn.05]] corrige les points faibles de GRPO et présente l'alternative qui consiste à distiller le raisonnement d'un modèle plus grand.

## L'idée en termes simples

Un enseignant donne à un élève un problème difficile et demande **quatre** tentatives différentes. Un correcteur note chacune juste ou fausse. Deux tentatives étaient justes, deux fausses. L'enseignant n'a pas besoin d'un « modèle de valeur » expert pour dire à l'élève ce que vaut une tentative typique : les quatre tentatives *sont* la référence. On félicite les deux qui ont fait mieux que la moyenne du groupe, on décourage les deux qui ont fait moins bien, et on recommence avec le problème suivant. Si les quatre étaient justes (ou toutes fausses), personne ne se démarque, et il n'y a rien à apprendre de ce problème aujourd'hui.

« Féliciter » signifie : rendre les mots exacts de cette tentative un peu plus probables la prochaine fois. « Décourager » signifie : les rendre un peu moins probables. Fais cela sur des milliers de problèmes et le modèle dérive vers les habitudes qui mènent à des réponses correctes, comme détailler les étapes, revérifier et encadrer la réponse.

Le chapitre 6 du livre oppose cela au RLHF ([[llm.05]]), où un **modèle de récompense appris** note les réponses en imitant les préférences humaines. Avec des **récompenses vérifiables (RLVR)**, la récompense vient d'un programme (le vérificateur de [[rsn.01]], un test unitaire, un compilateur) : elle est donc peu coûteuse, objective et impossible à embobiner.

## Des récompenses à un gradient

### Mise en place

- Prompt $x$ (un problème). Politique $\pi_\theta(y\mid x)$ : le LLM de poids $\theta$, qui génère une séquence de tokens $y=(y_1,\dots,y_T)$ avec la probabilité $\pi_\theta(y\mid x)=\prod_{t}\pi_\theta(y_t\mid x,y_{<t})$. Un $y$ échantillonné s'appelle un **rollout** ou une complétion.
- Récompense $r(x,y)$ : par exemple $1$ si la réponse encadrée est vérifiée comme correcte et $0$ sinon.
- Objectif : $J(\theta)=\mathbb{E}_{y\sim\pi_\theta}[r(x,y)]$, la récompense espérée. Nous voulons la faire monter.

### Le gradient de politique (rappel de [[rl.05]])

$J$ dépend de $\theta$ uniquement à travers la distribution d'échantillonnage, donc nous utilisons l'**astuce de la log-dérivée** $\nabla\pi=\pi\,\nabla\log\pi$ :

$$
\nabla_\theta J=\sum_y r(y)\,\nabla_\theta\pi_\theta(y)=\sum_y \pi_\theta(y)\,r(y)\,\nabla_\theta\log\pi_\theta(y)=\mathbb{E}_{y\sim\pi_\theta}\bigl[r(y)\,\nabla_\theta\log\pi_\theta(y)\bigr].
$$

On échantillonne des rollouts et on pondère le gradient de la log-probabilité de chacun par sa récompense : c'est REINFORCE. La log-probabilité d'une séquence entière est la **somme** des log-probabilités de ses tokens, donc chaque token d'un rollout est poussé avec le même poids.

### Pourquoi une baseline, et pourquoi un groupe

Les récompenses brutes sont toutes positives ou nulles : même un mauvais rollout serait poussé *vers le haut*. Le remède est de soustraire une **baseline** $b$ qui ne dépend pas du $y$ échantillonné :

$$
\mathbb{E}_y\bigl[b\,\nabla\log\pi(y)\bigr]=b\sum_y\nabla\pi(y)=b\,\nabla\sum_y\pi(y)=b\,\nabla 1=0,
$$

donc $\mathbb{E}[(r-b)\nabla\log\pi]=\nabla J$ : soustraire une baseline laisse le gradient inchangé en moyenne et réduit en général sa variance. PPO (voir [[rl.05]]) apprend un réseau de valeur $V_\phi(x)$ comme baseline : un second modèle de la taille d'un LLM à entraîner et à garder en mémoire. L'astuce de GRPO : **échantillonner un groupe de $G$ rollouts pour le même prompt et utiliser la récompense moyenne du groupe comme baseline.**

Point subtil (dérivé ici, absent du livre) : la moyenne du groupe contient $r_i$ elle-même. Écris $b=\frac1G\sum_jr_j$. Pour $j\ne i$, $r_j$ est indépendant de $y_i$, donc $\mathbb{E}[r_j\nabla\log\pi(y_i)]=\mathbb{E}[r_j]\cdot0=0$. Seul le terme $j=i$ survit, et
$$
\mathbb{E}\bigl[(r_i-b)\nabla\log\pi(y_i)\bigr]=\Bigl(1-\frac1G\Bigr)\nabla J .
$$
Ainsi la baseline de groupe laisse la direction du gradient exacte et ne fait que la multiplier par $(G-1)/G$.

### Avantage relatif au groupe

Le livre divise en plus par l'écart type du groupe. Pour les récompenses $r_1,\dots,r_G$ des rollouts d'un même prompt :

$$
A_i=\frac{r_i-\mu_r}{\sigma_r+\varepsilon},\qquad \mu_r=\frac1G\sum_j r_j,\quad \sigma_r=\sqrt{\frac1{G-1}\sum_j(r_j-\mu_r)^2},
$$

avec un petit $\varepsilon$ ($10^{-4}$) pour éviter la division par zéro. Propriétés :

- Les avantages somment à zéro, il y a donc toujours des poussées « vers le haut » et d'autres « vers le bas ». L'avantage moyen est un test de cohérence du code (il doit rester à 0).
- Si toutes les récompenses sont égales, $r_i-\mu_r=0$ pour tout $i$ : **aucun signal d'apprentissage.**
- Avec des récompenses binaires et $k$ rollouts corrects sur $G$, la variance échantillonnée vaut $\frac{k(G-k)}{G(G-1)}$, donc un rollout correct a $A=\frac{1-k/G}{\sigma}$ et un rollout faux $A=-\frac{k/G}{\sigma}$. Pour $[1,1,0,0]$ : $\sigma=\sqrt{1/3}=0.5774$, $A=\pm0.866$ (le code donne $0.8659$ à cause de $\varepsilon$). Un succès rare parmi huit ($k=1$) obtient $A=+2.474$ tandis que chaque échec n'a que $-0.354$ : **les succès rares sont fortement amplifiés.**

### La perte

La montée de gradient de politique sur $\frac1G\sum_iA_i\log\pi_\theta(y_i\mid x)$ est une descente de gradient sur

$$
\mathcal{L}_{\text{PG}}=-\frac1G\sum_{i=1}^{G}A_i\,\log\pi_\theta(y_i\mid x)=-\frac1G\sum_{i=1}^{G}A_i\sum_{t=1}^{T_i}\log\pi_\theta\bigl(y^{(i)}_t\mid x,y^{(i)}_{<t}\bigr).
$$

Les avantages sont traités comme des constantes (`detach` dans PyTorch) : on ne différencie qu'à travers les log-probabilités. Le signe moins est là parce que les optimiseurs minimisent.

*Somme ou moyenne sur les tokens ?* Le livre somme (objectif au niveau de la séquence). Moyenner diviserait la poussée de chaque rollout par sa longueur, un rééchelonnement involontaire dépendant de la longueur. La log-probabilité sommée est aussi plus négative pour les sorties longues, ce qui, comme le note le livre, favorise légèrement les réponses plus courtes à qualité égale.

### Clipping à la PPO et ancrage KL

L'entraînement réel effectue plusieurs pas d'optimisation sur le même lot de rollouts ; après le premier pas, la politique qui les a générés ($\pi_{\text{old}}$) n'est donc plus celle qu'on met à jour. Définissons le **ratio**
$\rho_i=\pi_\theta(y_i\mid x)/\pi_{\text{old}}(y_i\mid x)=\exp(\log\pi_\theta-\log\pi_{\text{old}})$, et remplaçons $A_i\log\pi_\theta$ par le surrogat avec clipping utilisé dans PPO (voir [[rl.05]]) :

$$
\mathcal{L}=-\frac1G\sum_i\min\bigl(\rho_iA_i,\ \operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\,A_i\bigr)+\beta\,\frac1G\sum_i\bigl(\log\pi_\theta(y_i\mid x)-\log\pi_{\text{ref}}(y_i\mid x)\bigr).
$$

Pourquoi cette forme :

- Pour $\rho=1$, le gradient de $\rho A$ par rapport à $\theta$ est $A\nabla\log\pi_\theta$, donc l'objectif sans clipping se réduit au gradient de politique ci-dessus. **Pour $\rho=1$, la valeur de la perte est 0** (les avantages somment à zéro) **mais le gradient ne l'est pas** : n'utilise pas la perte journalisée comme indicateur de progrès.
- Le $\min$ rend la mise à jour *pessimiste*. Si $A>0$ et $\rho>1+\epsilon$, le terme clippé est le plus petit et il est constant en $\theta$ : le gradient est nul, car ce rollout est déjà devenu plus probable que permis. Idem si $A<0$ et $\rho<1-\epsilon$. Mais si $A<0$ et $\rho>1+\epsilon$, le terme non clippé est le plus petit et le gradient passe (la politique a rendu plus probable un mauvais rollout et doit être corrigée).
- Le livre note que DeepSeek-R1 a utilisé un clip très faible ($\epsilon=10$), alors que $0.1$ à $0.2$ est typique ailleurs.
- Le dernier terme est une **pénalité de type KL** par rapport à un modèle de référence figé (le modèle de départ). Sa moyenne d'échantillon estime $\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})$ quand les rollouts viennent de $\pi_\theta$ ; elle décourage de s'éloigner du comportement d'origine. Elle coûte une seconde copie du modèle. Le chapitre 7 rapporte que cette forme simple, calculée sur des log-probabilités **sommées**, a déstabilisé l'entraînement dans l'expérience des auteurs ([[rsn.05]]) ; la recommandation courante pour les maths est de ne pas utiliser de KL du tout.

### Tout sur une image

```mermaid
flowchart LR
  P[prompt x] --> S[sample G rollouts]
  S --> R[verifier: rewards r_i]
  R --> A[advantages A_i = (r_i - mean) / std]
  S --> L[sequence log-probs]
  A --> G[GRPO loss]
  L --> G
  G --> U[backward, clip grad norm, AdamW step]
  U --> P
```

## Conception de la récompense

La récompense est toute la spécification de la tâche. D'après le livre :

- **Récompense de correction.** $1$ si la dernière réponse encadrée est vérifiée, sinon $0$. Exiger l'encadrement joue aussi le rôle de **récompense de format** implicite, car une réponse correcte non encadrée ne rapporte rien (pour les quatre rollouts d'exemple « boxed 83 », « the correct answer is boxed 83 », « the final answer is 83 » et « we get boxed 38 » à un problème dont la réponse est 83, les récompenses sont $[1,1,0,0]$ : la troisième est juste mais non encadrée).
- **Pas de récompense de processus.** Le livre indique que l'équipe de DeepSeek-R1 a essayé les modèles de récompense de processus (qui notent les étapes intermédiaires), les a trouvés peu concluants, et n'a utilisé que des récompenses sur la réponse finale.
- **Récompense de format explicite** (chapitre 7) : ajouter $+1$ si `<think>` apparaît avant `</think>`. Sans condition, elle peut dominer : une réponse fausse mais bien formatée rapporte $1$, autant qu'une réponse juste sans balises. Le livre a vu le modèle s'améliorer pendant une centaine de pas, puis se dégrader, avec des réponses plus courtes, et suggère un poids plus faible ou une récompense de format **conditionnelle** (le format ne compte que si la réponse est correcte). Dans l'exercice : juste + balises $=2$ ; faux + balises $=1$ sans condition mais $0$ avec condition.
- **Problèmes sans signal.** Si un problème est résolu par *tous* les rollouts ou par *aucun*, son groupe a des avantages nuls. Avec une probabilité de succès $p$ et $G$ rollouts, la probabilité de cela est $p^G+(1-p)^G$ : pour $p=0.2$ elle vaut $0.41$ à $G=4$ et $0.17$ à $G=8$ ; pour $p=0.05$ elle vaut $0.81$ à $G=4$. Des problèmes difficiles et un modèle faible donnent surtout des groupes vides ; davantage de rollouts par prompt et un curriculum de difficulté modérée aident. Le livre note qu'une récompense moyenne de $1.0$ signifie qu'il ne reste plus de signal : arrête-toi ou prends des problèmes plus difficiles.

## Ce qui s'est passé dans l'expérience du livre

Qwen3 0.6B base, 50 pas de GRPO, 8 rollouts d'au plus 512 tokens, taux d'apprentissage $10^{-5}$ avec AdamW, clip de la norme du gradient à 1.0, problèmes d'entraînement MATH excluant MATH-500 :

| modèle | précision MATH-500 | tokens générés en moyenne |
|---|---|---|
| base | 15.2% | 78.85 |
| variante de raisonnement (référence) | 48.2% | 1369.79 |
| base + 50 pas GRPO (sans KL) | 47.4% | 586.11 |

Cinquante pas de RL ont rapproché le modèle de base de la variante de raisonnement officielle, avec des réponses 7,4 fois plus longues que la base (le raisonnement est écrit) mais moins de moitié aussi longues que la référence. Le livre prévient qu'un entraînement plus long est instable et peut même nuire, ce qui motive [[rsn.05]]. (Il s'agit d'une expérience sur un modèle ; des écarts d'un point ou deux sont dans le bruit discuté en [[rsn.01]].)

Le calcul est important : le tableau de l'annexe E du livre indique 68 minutes (H100) pour le script sans batch et 19 minutes avec 4 rollouts en batch, pour 43 à 45 Go de mémoire.

## Exemple détaillé

Quatre rollouts pour un problème dont la réponse est 83 ont pour récompenses $[1,1,0,0]$.

1. $\mu=0.5$, $\sigma=\sqrt{\frac13}=0.5774$ (écart type d'échantillon).
2. Avantages : $\bigl(\tfrac{0.5}{0.5774+10^{-4}}\bigr)=0.8659$ pour les deux premiers, $-0.8659$ pour les autres.
3. Supposons que les log-probabilités des séquences soient $\ell=[-7.9243,-20.1546,-16.6130,-23.3677]$ (les valeurs que le livre utilise pour illustrer la perte). Alors $\mathcal{L}_{\text{PG}}=-\frac14\sum_iA_i\ell_i=-\frac14\bigl(0.8659\cdot(-28.0789)-0.8659\cdot(-39.9807)\bigr)=-2.5764$. Le signe n'est pas significatif en soi ; ce qui compte est le gradient, qui augmente $\ell_1,\ell_2$ et diminue $\ell_3,\ell_4$.
4. Dans la politique jouet de l'exercice ($T=3$, $V=4$), les quatre mêmes avantages produisent un gradient analytique qui concorde avec les différences finies centrées à mieux que $10^{-6}$, y compris quand deux des quatre rollouts sont clippés et qu'un terme KL est présent.

## Implémentation

```python
import numpy as np

def group_advantages(r, eps=1e-4):
    r = np.asarray(r, float)
    return (r - r.mean()) / (r.std(ddof=1) + eps)

def grpo_loss(logp, old_logp, ref_logp, adv, clip_eps=0.2, beta=0.0):
    ratio = np.exp(logp - old_logp)
    obj = np.minimum(ratio * adv, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    return -obj.mean() + beta * np.mean(logp - ref_logp)
```

En PyTorch : `logp = selected_logprobs.sum()` par rollout avec les gradients activés, `adv` et `old_logp` détachés, `loss.backward()`, `clip_grad_norm_(params, 1.0)`, `optimizer.step()`. Les rollouts sont échantillonnés sous `torch.no_grad` (pas `inference_mode`, qui bloque l'usage ultérieur d'autograd). La version de production suit aussi la récompense, la longueur des réponses, la moyenne et l'écart type des avantages, et l'entropie ([[rsn.05]]), et met les rollouts en batch. L'exercice tabulaire ne diffère d'un vrai LLM que par la façon de calculer $\log\pi_\theta$, pas par la perte.

## Idées reçues

- « GRPO a besoin d'un modèle de récompense. » Pas pour les tâches vérifiables ; le vérificateur est le modèle de récompense.
- « La moyenne du groupe est une baseline non biaisée, donc il n'y a pas de problème. » Elle est seulement mise à l'échelle par $(G-1)/G$ ; et la division supplémentaire par l'écart type modifie les poids entre problèmes ([[rsn.05]]).
- « Une perte proche de 0 signifie que l'entraînement est bloqué. » Avec une seule mise à jour par lot, le ratio vaut 1, la valeur de la perte est 0 par construction, et pourtant le gradient ne l'est pas.
- « Si la récompense moyenne est 1.0, parfait, on continue. » Il ne reste aucun signal ; utilise des problèmes plus difficiles.
- « Une récompense de format ne coûte rien. » Elle peut devenir ce qui est optimisé.

## Points clés

- RLVR : récompense = verdict d'un programme ; GRPO échantillonne un groupe par prompt et utilise le groupe comme baseline.
- $A_i=(r_i-\mu)/(\sigma+\varepsilon)$ ; nul quand toutes les récompenses coïncident ; les succès rares reçoivent de grands avantages positifs.
- Perte : log-probabilités sommées pondérées par l'avantage, avec clipping à la PPO par rapport à $\pi_{\text{old}}$ et un terme KL optionnel par rapport à $\pi_{\text{ref}}$.
- Vérifie ton gradient numériquement ; surveille la récompense, la longueur, les statistiques des avantages, pas la perte brute.
- Concevoir la récompense, c'est spécifier la tâche : attention aux termes de format et aux groupes sans signal.

## Explique-le toi-même

1. Pourquoi GRPO peut-il s'entraîner sans réseau de valeur, et quel en est le prix ?
2. Explique avec tes mots pourquoi la perte vaut exactement zéro au premier pas alors que le gradient ne l'est pas.

## Pour aller plus loin

- Chapitre 6 du livre (algorithme), chapitre 7.2-7.6 (suivi, clipping, KL, récompense de format) ; gradients de politique dans [[rl.05]] ; modèles de récompense et RLHF dans [[llm.05]].
- Articles DeepSeekMath et DeepSeek-R1 (origine de GRPO), cités par les notebooks mais non ouverts pour cette leçon.

## Critères de maîtrise

- Je sais dériver $\nabla J=\mathbb{E}[r\nabla\log\pi]$ et montrer qu'une baseline ne le biaise pas.
- Je sais calculer à la main des avantages relatifs au groupe et dire quand ils s'annulent.
- Je sais écrire la perte avec clipping et régularisation KL et dire quels rollouts reçoivent un gradient nul à cause du clipping.
- Je sais vérifier numériquement le gradient d'une perte de RL.
- Je sais expliquer deux écueils des récompenses de format et des groupes sans signal.
