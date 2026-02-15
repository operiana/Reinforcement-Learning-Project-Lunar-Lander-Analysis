> Oscar Perianayagassamy, M2 IASD App., Université Paris-Dauphine

# Lunar Lander : Analyse comparative DQN, A2C & PPO sur différents taux d'apprentissage

Ce projet étudie l'impact des **Learning Rates** sur la convergence des algorithmes de DeepRL : `DQN`, `A2C` et `PPO`. Ces algorithmes sont entrainés sur 6 variantes du problème `Lunar Lander` : 
- Environnement d'actions discret : variante classique (sans modification des arguments par défaut de `Gymnasium`), variante venteuse et variante zéro-gravité venteuse.
- Environnement d'actions continue : variante classique, variante venteuse et variante zéro-gravité venteuse.

Tous les détails se trouvent dans le fichier <a href="rapport.pdf">rapport.pdf</a>.

## Quelques exemples de PPO 
Voici les performances de nos agents sur la variante **Windy Continuous**.

### `PPO` / actions discrètes (`VANILLA`) 

version `CLASSIC`

<img src="gifs/classic/vanilla/lunar_lander_PPO_0.0005623_200K_classic-vanilla/run_0_reward_244.585.gif" width="200" /><img src="gifs/classic/vanilla/lunar_lander_PPO_0.0005623_200K_classic-vanilla/run_1_reward_276.513.gif" width="200" /><img src="gifs/classic/vanilla/lunar_lander_PPO_0.0005623_200K_classic-vanilla/run_2_reward_251.156.gif" width="200" />

version `WINDY`

<img src="gifs/windy/vanilla/lunar_lander_PPO_0.0005623_200K_windy-vanilla/run_1_reward_-149.501.gif" width="200" /><img src="gifs/windy/vanilla/lunar_lander_PPO_0.0005623_200K_windy-vanilla/run_3_reward_273.010.gif" width="200" /><img src="gifs/windy/vanilla/lunar_lander_PPO_0.0005623_200K_windy-vanilla/run_4_reward_257.840.gif" width="200" />

version `ZERO-GRAVITY-WINDY`

<img src="gifs/zero-gravity-windy/vanilla/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-vanilla/run_2_reward_-284.396.gif" width="200" /><img src="gifs/zero-gravity-windy/vanilla/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-vanilla/run_3_reward_-77.860.gif" width="200" /><img src="gifs/zero-gravity-windy/vanilla/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-vanilla/run_4_reward_39.936.gif" width="200" />

### `PPO` / actions continues (`CONTINUOUS`) 

version `CLASSIC`

<img src="gifs/classic/continuous/lunar_lander_PPO_0.0005623_200K_classic-continuous/run_0_reward_-36.958.gif" width="200" /><img src="gifs/classic/continuous/lunar_lander_PPO_0.0005623_200K_classic-continuous/run_1_reward_8.375.gif" width="200" /><img src="gifs/classic/continuous/lunar_lander_PPO_0.0005623_200K_classic-continuous/run_4_reward_185.516.gif" width="200" />

version `WINDY`

<img src="gifs/windy/continuous/lunar_lander_PPO_0.0005623_200K_windy-continuous/run_2_reward_-8.394.gif" width="200" /><img src="gifs/windy/continuous/lunar_lander_PPO_0.0005623_200K_windy-continuous/run_3_reward_102.594.gif" width="200" /><img src="gifs/windy/continuous/lunar_lander_PPO_0.0005623_200K_windy-continuous/run_1_reward_233.715.gif" width="200" />

version `ZERO-GRAVITY-WINDY`

<img src="gifs/zero-gravity-windy/continuous/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-continuous/run_3_reward_-658.006.gif" width="200" /><img src="gifs/zero-gravity-windy/continuous/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-continuous/run_2_reward_-109.591.gif" width="200" /><img src="gifs/zero-gravity-windy/continuous/lunar_lander_PPO_0.0005623_200K_zero-gravity-windy-continuous/run_4_reward_-83.965.gif" width="200" />


## 2. Protocole Expérimental
Nous avons testé 5 Learning Rates sur une échelle logarithmique : 
$\sim 10^{-7}, 10^{-6}, 10^{-4}, 10^{-3}, 10^{-2}$.



## 3. Installation
Le projet a été développé sous `Python3.13`.

```bash
git clone [https://github.com/operiana/Reinforcement-Learning-Project-Lunar-Lander-Analysis.git](https://github.com/operiana/Reinforcement-Learning-Project-Lunar-Lander-Analysis.git)
pip install -r requirements.txt