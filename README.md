> Oscar Perianayagassamy, M2 IASD App., Université Paris-Dauphine

# Lunar Lander : Analyse comparative DQN, A2C & PPO sur différents taux d'apprentissage

Ce projet étudie l'impact des **Learning Rates** sur la convergence des algorithmes de DeepRL : `DQN`, `A2C` et `PPO`. Ces algorithmes sont entrainés sur 6 variantes du problème `Lunar Lander` : 
- Environnement d'actions discret : variante classique (sans modification des arguments par défaut de `Gymnasium`), variante venteuse et variante zéro-gravité venteuse.
- Environnement d'actions continue : variante classique, variante venteuse et variante zéro-gravité venteuse.

Tous les détails se trouvent dans le fichier <a href="rapport.pdf">rapport.pdf</a>.

## Quelques exemples des meilleurs modèles PPO entrainés
Voici les performances de nos agents sur les différentes variantes étudiées.

### `PPO` / actions discrètes (`VANILLA`) 

version `CLASSIC`

<img src="gifs/classic/vanilla/lunar_lander_PPO_0.0005623_200K_classic-vanilla/run_0_reward_209.gif" width="200" /><img src="gifs/classic/vanilla/lunar_lander_PPO_0.0005623_200K_classic-vanilla/run_0_reward_247.gif" width="200" /><img src="gifs/classic/vanilla/lunar_lander_PPO_0.0005623_200K_classic-vanilla/run_0_reward_238.gif" width="200" /><img src="gifs/classic/vanilla/lunar_lander_PPO_0.0005623_200K_classic-vanilla/run_0_reward_248.gif" width="200" /><img src="gifs/classic/vanilla/lunar_lander_PPO_0.0005623_200K_classic-vanilla/run_0_reward_245.gif" width="200" />

version `WINDY`

<img src="gifs/windy/vanilla/lunar_lander_PPO_0.0005623_200K_windy-vanilla/run_1_reward_245.gif" width="200" /><img src="gifs/windy/vanilla/lunar_lander_PPO_0.0005623_200K_windy-vanilla/run_1_reward_240.gif" width="200" /><img src="gifs/windy/vanilla/lunar_lander_PPO_0.0005623_200K_windy-vanilla/run_1_reward_241.gif" width="200" /><img src="gifs/windy/vanilla/lunar_lander_PPO_0.0005623_200K_windy-vanilla/run_1_reward_212.gif" width="200" /><img src="gifs/windy/vanilla/lunar_lander_PPO_0.0005623_200K_windy-vanilla/run_1_reward_133.gif" width="200" />

version `ZERO-GRAVITY-WINDY`

<img src="gifs/zero-gravity-windy/vanilla/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-vanilla/run_2_reward_-166.gif" width="200" /><img src="gifs/zero-gravity-windy/vanilla/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-vanilla/run_2_reward_-346.gif" width="200" /><img src="gifs/zero-gravity-windy/vanilla/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-vanilla/run_2_reward_-25.gif" width="200" /><img src="gifs/zero-gravity-windy/vanilla/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-vanilla/run_2_reward_-131.gif" width="200" /><img src="gifs/zero-gravity-windy/vanilla/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-vanilla/run_2_reward_-74.gif" width="200" />

### `PPO` / actions continues (`CONTINUOUS`) 

version `CLASSIC`

<img src="gifs/classic/continuous/lunar_lander_PPO_0.0005623_200K_classic-continuous/run_0_reward_-8.gif" width="200" /><img src="gifs/classic/continuous/lunar_lander_PPO_0.0005623_200K_classic-continuous/run_0_reward_138.gif" width="200" /><img src="gifs/classic/continuous/lunar_lander_PPO_0.0005623_200K_classic-continuous/run_0_reward_-40.gif" width="200" /><img src="gifs/classic/continuous/lunar_lander_PPO_0.0005623_200K_classic-continuous/run_0_reward_114.958.gif" width="200" /><img src="gifs/classic/continuous/lunar_lander_PPO_0.0005623_200K_classic-continuous/run_0_reward_122.958.gif" width="200" />

version `WINDY`

<img src="gifs/windy/continuous/lunar_lander_PPO_0.0005623_200K_windy-continuous/run_2_reward_34.gif" width="200" /><img src="gifs/windy/continuous/lunar_lander_PPO_0.0005623_200K_windy-continuous/run_2_reward_247.gif" width="200" /><img src="gifs/windy/continuous/lunar_lander_PPO_0.0005623_200K_windy-continuous/run_2_reward_-12.gif" width="200" /><img src="gifs/windy/continuous/lunar_lander_PPO_0.0005623_200K_windy-continuous/run_2_reward_41.gif" width="200" /><img src="gifs/windy/continuous/lunar_lander_PPO_0.0005623_200K_windy-continuous/run_2_reward_-3.gif" width="200" />

version `ZERO-GRAVITY-WINDY`

<img src="gifs/zero-gravity-windy/continuous/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-continuous/run_3_reward_-120.gif" width="200" /><img src="gifs/zero-gravity-windy/continuous/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-continuous/run_3_reward_-124.gif" width="200" /><img src="gifs/zero-gravity-windy/continuous/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-continuous/run_3_reward_-230.gif" width="200" /><img src="gifs/zero-gravity-windy/continuous/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-continuous/run_3_reward_-194.gif" width="200" /><img src="gifs/zero-gravity-windy/continuous/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-continuous/run_3_reward_-364.gif" width="200" />


## 2. Protocole Expérimental
Nous avons testé 5 Learning Rates sur une échelle logarithmique : 
$\sim 10^{-7}, 10^{-6}, 10^{-4}, 10^{-3}, 10^{-2}$.



## 3. Installation
Le projet a été développé sous `Python3.13`.

```bash
git clone [https://github.com/operiana/Reinforcement-Learning-Project-Lunar-Lander-Analysis.git](https://github.com/operiana/Reinforcement-Learning-Project-Lunar-Lander-Analysis.git)
pip install -r requirements.txt
