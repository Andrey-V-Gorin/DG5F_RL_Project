# DG5F RL Project: UR10e + DG-5F + ResNet18 + SAC/PPO

Проект предназначен для обучения и сравнения алгоритмов SAC и PPO в задаче захвата объектов связкой **UR10e + DELTO DG-5F** в MuJoCo. Основной файл проекта: `train_dg5f_ur10e_resnet18_sac_ppo.py`.

## 1. Структура проекта

Ожидаемая структура каталога:

```text
DG5F_RL_Project/
├── train_dg5f_ur10e_resnet18_sac_ppo.py
├── DG5F_Preflight_Checks.ipynb
├── DG5F_Research.ipynb
├── README.md
├── tesollo_dg5f_mujoco-main/
├── results/
├── checkpoints/
├── figures/
├── videos/
├── generated_scenes/
├── logs/
├── configs/
└── preflight_reports/
```

Каталоги `results`, `checkpoints`, `figures`, `videos`, `generated_scenes`, `logs`, `configs`, `preflight_reports` создаются и наполняются автоматически.

## 2. Быстрая проверка окружения

```bash
python -c "import torch, mujoco, gymnasium, stable_baselines3; print('OK')"
python -c "import pinocchio as pin; print(pin.__version__)"
```

## 3. Pre-flight диагностика

Перед обучением откройте `DG5F_Preflight_Checks.ipynb` и выполните ячейки сверху вниз.

Notebook вызывает функцию:

```python
run_preflight_checks(config)
```

Она проверяет создание среды, reset, положение объекта, рендер камер, action space, изменение reward, контакты, вредные контакты и корректность выполнения эпизода.

Результаты сохраняются в:

```text
preflight_reports/preflight_report_<object>.csv
figures/preflight_<object>_<camera>.png
```

## 4. Быстрый тест

В `DG5F_Research.ipynb` установите:

```python
MODE = "quick"
OBJECT_SHAPES = ["cube"]
```

Быстрый режим использует малое число шагов и нужен только для проверки, что проект запускается, сохраняет модели, метрики, графики и GIF.

## 5. Полный эксперимент

В `DG5F_Research.ipynb` установите:

```python
MODE = "full"
OBJECT_SHAPES = ["cube", "cylinder", "sphere"]
```

Полный режим запускает обучение SAC и PPO, рассчитывает метрики и сохраняет результаты.

## 6. Запуск из командной строки

```bash
python train_dg5f_ur10e_resnet18_sac_ppo.py   --object-shapes cube cylinder sphere   --timesteps-sac 150000   --timesteps-ppo 150000   --eval-episodes 10   --max-episode-steps 200
```

Без GIF:

```bash
python train_dg5f_ur10e_resnet18_sac_ppo.py --object-shapes cube --no-gif
```

## 7. Что сохраняется

```text
results/sac_vs_ppo_metrics_ur10e_dg5f.csv
checkpoints/sac_ur10e_dg5f_<object>.zip
checkpoints/ppo_ur10e_dg5f_<object>.zip
videos/rollout_<algorithm>_<object>_two_cameras.gif
figures/*.png
logs/experiment_log.txt
logs/metrics_summary.txt
configs/config_<object>.json
configs/last_experiment_config.json
preflight_reports/preflight_report_<object>.csv
generated_scenes/scene_ur10e_dg5f_<object>.xml
```

## 8. Основные функции

`DG5F_Research.ipynb` может вызывать все ключевые функции основного файла:

```python
ensure_output_dirs(config)
save_config_snapshot(config, object_shapes)
run_preflight_checks(config)
train_and_evaluate(config)
train_and_evaluate_multishape(config, object_shapes)
evaluate_policy_custom(model, config, algorithm_name, object_shape)
save_rollout_gif(model, config, algorithm_name, object_shape)
create_metric_figures(metrics, config)
print_metrics(metrics)
```

## 9. Камеры GIF

GIF формируется одновременно с двух камер. Камеры имеют общие параметры удаленности и высоты:

```python
camera_distance
camera_height
```

Для каждой камеры отдельно регулируются:

```python
camera_1_lookat_x, camera_1_lookat_y, camera_1_azimuth, camera_1_elevation
camera_2_lookat_x, camera_2_lookat_y, camera_2_azimuth, camera_2_elevation
```

## 10. Рекомендуемый порядок работы

1. Запустить `DG5F_Preflight_Checks.ipynb`.
2. Исправить конфигурацию, если есть `FAIL`.
3. Запустить `DG5F_Research.ipynb` в режиме `quick`.
4. Проверить GIF и метрики.
5. Запустить `DG5F_Research.ipynb` в режиме `full`.
