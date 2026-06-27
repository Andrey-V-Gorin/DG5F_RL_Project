"""
Проект ВКР: управление UR10e + DG-5F на основе ResNet18, SAC и PPO.

Файл является единой рабочей точкой запуска экспериментов:
- загружает реальную MJCF-модель UR10e + DG-5F из Tesollo;
- формирует отдельную MuJoCo-сцену задачи захвата;
- управляет одновременно 6 суставами UR10e и 20 суставами DG-5F;
- использует визуальные признаки ResNet18 в составе состояния;
- обучает и сравнивает SAC и PPO;
- считает Success Rate, Average Reward, MAE, MSE, RMSE, ITAE,
  время обучения и время формирования управляющего воздействия;
- сохраняет GIF rollout одновременно с двух камер.

Важно:
1. В проекте используется собственная Gymnasium-обертка над MuJoCo.
   Это сделано для прозрачного контроля reward, success, контактов, камер и метрик.
2. При этом используется реальная модель Tesollo: tesollo_dg5f_mujoco-main/robot/ur10edg5f.xml и все связанные assets UR10e + DG-5F.
3. Объект манипулирования добавляется в автоматически создаваемую сцену.
"""

from __future__ import annotations

# =============================================================================
# Блок 1. Импорты стандартной библиотеки
# =============================================================================

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# =============================================================================
# Блок 2. Импорты научных и ML-библиотек
# =============================================================================

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

import mujoco

import torch
import torch.nn as nn
from PIL import Image

try:
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision import transforms
except Exception as exc:  # pragma: no cover
    raise ImportError("Не удалось импортировать torchvision. Установите torch и torchvision.") from exc

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# =============================================================================
# Блок 3. Конфигурация эксперимента
# =============================================================================

@dataclass
class ExperimentConfig:
    """Единая конфигурация эксперимента UR10e + DG-5F.
    Все параметры вынесены в dataclass, чтобы их можно было менять из Notebook, CLI или конфигурационных файлов без изменения логики среды.
    """

    # -------------------------------------------------------------------------
    # Пути проекта и исходной модели Tesollo
    # -------------------------------------------------------------------------
    project_root: str = "."
    tesollo_root: str = "tesollo_dg5f_mujoco-main"

    # -------------------------------------------------------------------------
    # Параметры изображения и ResNet18
    # -------------------------------------------------------------------------
    image_size: int = 224
    use_pretrained_resnet: bool = True
    freeze_resnet: bool = True

    # -------------------------------------------------------------------------
    # Параметры MuJoCo
    # -------------------------------------------------------------------------
    sim_timestep: float = 0.001
    control_hz: float = 20.0
    max_episode_steps: int = 200
    render_mode: str = "rgb_array"

    # -------------------------------------------------------------------------
    # Параметры объекта манипулирования
    # -------------------------------------------------------------------------
    object_shape: str = "cube"  # "cube", "cylinder", "sphere"
    object_name: str = "grasp_object"
    object_geom_name: str = "grasp_object_geom"
    object_joint_name: str = "grasp_object_joint"
    object_size: float = 0.025
    object_mass: float = 0.15

    # Начальная позиция объекта в системе координат сцены UR10e.
    # Для UR10e + DG-5F объект должен находиться на столе в зоне достижимости.
    object_initial_pos_x: float = -0.15
    object_initial_pos_y: float = -0.55
    object_initial_pos_z: float = 0.20
    object_random_xy: float = 0.01

    # -------------------------------------------------------------------------
    # Начальная поза UR10e + DG-5F
    # Первые 6 значений — UR10e, следующие 20 — DG-5F.
    # -------------------------------------------------------------------------
    initial_ur_qpos_0: float = 1.42010733
    initial_ur_qpos_1: float = -1.74898752
    initial_ur_qpos_2: float = 2.36328641
    initial_ur_qpos_3: float = -2.17431840
    initial_ur_qpos_4: float = -1.57146472
    initial_ur_qpos_5: float = -0.14989266

    # -------------------------------------------------------------------------
    # Масштаб действий.
    # Action space агента нормирован в [-1, 1].
    # В среде действия интерпретируются как малые приращения целевых положений.
    # -------------------------------------------------------------------------
    ur_action_delta: float = 0.035
    finger_action_delta: float = 0.070

    # -------------------------------------------------------------------------
    # Ограничения нормированных действий пальцев DG-5F.
    # UR10e сохраняет полный диапазон [-1, 1], а для пальцев вводится
    # ограничение, чтобы агент не использовал сильное переразгибание пальцев
    # в направлении, противоположном естественному закрытию захвата.
    # -------------------------------------------------------------------------
    finger_action_min: float = 0.00
    finger_action_max: float = 1.00

    # -------------------------------------------------------------------------
    # Reward shaping
    # -------------------------------------------------------------------------
    # Reach Reward считается не как награда за факт близости к объекту, а как награда за уменьшение расстояния между end-effector и объектом:
    # reach_reward = reach_progress_weight * (previous_tracking_error - tracking_error).
    # Если робот приблизился к объекту, компонент положительный; если удалился — отрицательный.
    distance_weight: float = 4.0  # сохранено для совместимости старых конфигураций; в новой reward-логике не используется как основной reach-компонент
    reach_progress_weight: float = 100.0

    # Multi-Finger Contact Reward: награда начисляется не за одиночное касание, а за вовлечение нескольких пальцев в контакт с объектом.
    contact_reward: float = 10.0

    # Lift Reward и Success Reward усилены, чтобы агенту было выгодно не просто касаться объекта, а поднимать и успешно удерживать его.
    lift_reward: float = 100.0
    success_reward: float = 500.0

    # Hold Reward и Stable Grasp Reward начисляются на каждом шаге, если объект поднят и удерживается несколькими пальцами.
    hold_reward: float = 2.0
    stable_grasp_reward: float = 5.0

    # Time Penalty штрафует каждый шаг эпизода, чтобы агент учился выполнять захват быстрее.
    time_penalty: float = 0.01

    # Drop Penalty штрафует потерю объекта после того, как он уже был поднят.
    drop_penalty: float = 200.0
    drop_lift_threshold: float = 0.020
    drop_height_threshold: float = 0.005

    action_penalty_weight: float = 0.002
    velocity_penalty_weight: float = 0.001

    # -------------------------------------------------------------------------
    # Штрафы за плохую физическую стратегию
    # -------------------------------------------------------------------------
    object_escape_penalty: float = 80.0
    workspace_radius: float = 0.16
    palm_contact_penalty: float = 25.0

    # Границы стола (с учетом размеров из XML)
    table_x_min: float = -0.60
    table_x_max: float = 0.50
    table_y_min: float = -0.95
    table_y_max: float = -0.25

    # Допустимый выезд кисти за край стола
    table_margin: float = 0.05

    # Штраф за контакт объекта с нерабочими частями кисти или манипулятора.
    # Он применяется только если такой контакт не сопровождается полезным контактом пальцев и положительным подъемом объекта.
    bad_contact_penalty: float = 40.0
    bad_contact_lift_threshold: float = 0.005
    bad_contact_finger_threshold: float = 1.0

    # Штраф за переразгибание пальцев DG-5F.
    # Если суставы пальцев уходят ниже безопасного нижнего порога, reward уменьшается пропорционально величине переразгибания.
    finger_hyperextension_penalty: float = 10.0
    finger_safe_lower_limit: float = -0.05

    # -------------------------------------------------------------------------
    # Критерии успеха
    # -------------------------------------------------------------------------
    success_height_delta: float = 0.035
    success_distance_threshold: float = 0.080
    min_success_fingers: float = 2.0

    # -------------------------------------------------------------------------
    # Обучение SAC/PPO
    # -------------------------------------------------------------------------
    total_timesteps_sac: int = 10_000
    total_timesteps_ppo: int = 10_000
    learning_rate: float = 3e-4
    batch_size: int = 128
    buffer_size: int = 200_000
    gamma: float = 0.99
    tau: float = 0.005
    seed: int = 42
    device: str = "auto"
    progress_bar: bool = False

    # -------------------------------------------------------------------------
    # Продолжение обучения из сохраненных моделей
    # -------------------------------------------------------------------------
    # Если resume_training=True, train_and_evaluate() пытается загрузить уже сохраненные модели SAC/PPO и продолжить обучение, а не начинать с нуля.
    # Если путь к checkpoint не задан явно, используется стандартный путь:
    # checkpoints/sac_ur10e_dg5f_<object_shape>.zip и checkpoints/ppo_ur10e_dg5f_<object_shape>.zip.
    resume_training: bool = False
    resume_sac_checkpoint: str = ""
    resume_ppo_checkpoint: str = ""

    # При продолжении обучения обычно нужно False, чтобы счетчик шагов в логах продолжался, а не начинался снова с нуля.
    reset_num_timesteps_on_resume: bool = False

    # Суффикс для сохранения дообученных моделей. Если оставить пустым, файл будет сохранен как обычный
    # sac_ur10e_dg5f_<object_shape>.zip / ppo_ur10e_dg5f_<object_shape>.zip, то есть старая модель будет перезаписана.
    resumed_model_suffix: str = "continued"

    # -------------------------------------------------------------------------
    # Архитектура Actor/Critic в Stable-Baselines3
    # -------------------------------------------------------------------------
    actor_hidden_1: int = 512
    actor_hidden_2: int = 256
    actor_hidden_3: int = 256
    critic_hidden_1: int = 512
    critic_hidden_2: int = 256
    critic_hidden_3: int = 256

    # -------------------------------------------------------------------------
    # Оценка качества
    # -------------------------------------------------------------------------
    eval_episodes: int = 10

    # -------------------------------------------------------------------------
    # Выходные каталоги
    # -------------------------------------------------------------------------
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    figures_dir: str = "figures"
    videos_dir: str = "videos"
    generated_scenes_dir: str = "generated_scenes"
    logs_dir: str = "logs"
    configs_dir: str = "configs"
    preflight_reports_dir: str = "preflight_reports"

    # -------------------------------------------------------------------------
    # Настройки GIF и камер.
    # Камера 1 и камера 2 имеют одинаковые distance и lookat_z.
    # Регулируются azimuth/elevation/lookat_x/lookat_y/distance/height.
    # -------------------------------------------------------------------------
    save_gif: bool = True
    gif_fps: int = 20
    gif_width: int = 480
    gif_height: int = 360
    camera_distance: float = 0.70
    camera_height: float = 0.32

    camera_1_name: str = "front"
    camera_1_lookat_x: float = -0.15
    camera_1_lookat_y: float = -0.55
    camera_1_azimuth: float = 180.0
    camera_1_elevation: float = -18.0

    camera_2_name: str = "palm"
    camera_2_lookat_x: float = -0.15
    camera_2_lookat_y: float = -0.55
    camera_2_azimuth: float = 90.0
    camera_2_elevation: float = -18.0

# =============================================================================
# Блок 4. Служебные функции путей
# =============================================================================

    """Ожидаемая структура:
    DG5F_RL_Project/
    ├── train_dg5f_ur10e_resnet18_sac_ppo.py
    └── tesollo_dg5f_mujoco-main/
    """

def resolve_project_root(config: ExperimentConfig) -> Path:
    """Возвращает абсолютный путь к корню проекта.
    Если project_root равен ".", используется каталог, в котором лежит данный Python-файл.
    """

    if config.project_root and config.project_root != ".":
        return Path(config.project_root).expanduser().resolve()

    return Path(__file__).resolve().parent


def resolve_tesollo_root(config: ExperimentConfig) -> Path:
    """Возвращает путь к каталогу tesollo_dg5f_mujoco-main."""

    project_root = resolve_project_root(config)

    candidate = Path(config.tesollo_root).expanduser()
    if candidate.is_absolute():
        tesollo_root = candidate
    else:
        tesollo_root = project_root / candidate

    if not tesollo_root.exists():
        raise FileNotFoundError(
            "Не найден каталог Tesollo: "
            f"{tesollo_root}. Проверьте параметр tesollo_root."
        )

    return tesollo_root.resolve()

def ensure_output_dirs(config: ExperimentConfig) -> None:
    """Создает выходные каталоги для результатов, моделей, рисунков и видео."""

    project_root = resolve_project_root(config)

    for directory in [
        config.results_dir,
        config.checkpoints_dir,
        config.figures_dir,
        config.videos_dir,
        config.generated_scenes_dir,
        config.logs_dir,
        config.configs_dir,
        config.preflight_reports_dir,
    ]:
        (project_root / directory).mkdir(parents=True, exist_ok=True)

def timestamp_string() -> str:
    """Возвращает компактную временную метку для имен файлов эксперимента."""

    return time.strftime("%Y%m%d_%H%M%S")

def append_log(config: ExperimentConfig, message: str) -> None:
    """Добавляет строку в текстовый лог эксперимента.

    Лог хранится в каталоге logs/ и позволяет восстановить ход запуска:
    какие объекты обучались, когда началось и завершилось обучение, куда сохранены модели, метрики, GIF и рисунки.
    """

    ensure_output_dirs(config)

    log_path = resolve_project_root(config) / config.logs_dir / "experiment_log.txt"
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"

    with log_path.open("a", encoding="utf-8") as file:
        file.write(line)

def save_config_snapshot(
    config: ExperimentConfig,
    object_shapes: Optional[Iterable[str]] = None,
    filename: Optional[str] = None,
) -> Path:
    """Сохраняет фактическую конфигурацию эксперимента в каталог configs/.

    В файл записываются все поля ExperimentConfig, список объектов запуска,
    а также абсолютный путь проекта. Это делает результаты воспроизводимыми
    и позволяет приложить конфигурацию к ВКР вместе с метриками.
    """

    ensure_output_dirs(config)

    project_root = resolve_project_root(config)
    configs_dir = project_root / config.configs_dir

    if filename is None:
        filename = f"experiment_config_{timestamp_string()}.json"

    output_path = configs_dir / filename

    payload = {
        "project_root_resolved": str(project_root),
        "object_shapes": list(object_shapes) if object_shapes is not None else [config.object_shape],
        "config": asdict(config),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)

    append_log(config, f"Сохранена конфигурация эксперимента: {output_path}")
    return output_path


def save_metrics_log(config: ExperimentConfig, metrics: List[Dict[str, float]]) -> Path:
    """Сохраняет человекочитаемый лог итоговых метрик в logs/."""

    ensure_output_dirs(config)

    output_path = resolve_project_root(config) / config.logs_dir / "metrics_summary.txt"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Итоговые метрики эксперимента UR10e + DG-5F\n")
        file.write("=" * 80 + "\n")

        for row in metrics:
            file.write(f"Объект: {row.get('object')} | Алгоритм: {row.get('algorithm')}\n")
            for key, value in row.items():
                if key in {"object", "algorithm"}:
                    continue
                file.write(f"{key}: {value}\n")
            file.write("-" * 80 + "\n")

    append_log(config, f"Сохранен текстовый отчет метрик: {output_path}")
    return output_path


def run_preflight_checks(config: ExperimentConfig) -> Path:
    """Выполняет полный pre-flight контроль среды до запуска обучения.

    Функция переносит в основной Python-файл все проверки, ранее размещенные в
    DG5F_Preflight_Checks.ipynb. Это нужно, чтобы диагностика и обучение
    использовали одну и ту же реализацию среды, reward, success, контактов,
    камер и action space.

    Проверяются:
    - наличие проекта, каталога Tesollo и ключевых XML-файлов;
    - импорт ключевых библиотек;
    - создание сгенерированной сцены;
    - создание среды, reset(), step();
    - корректность action_space и observation_space;
    - расположение объекта относительно end-effector;
    - получение непустого RGB-изображения с камер;
    - влияние действий на суставные координаты;
    - отсутствие взрывного масштаба действия;
    - непостоянность reward;
    - возможность обнаружения контакта scripted-действиями;
    - возможность success-логики вернуть True в искусственно успешном состоянии.

    Итоги сохраняются в:
    - preflight_reports/preflight_report_<object>.csv;
    - preflight_reports/preflight_report_<object>.txt;
    - figures/preflight_<object>_<camera>.png;
    - figures/preflight_reward_trace_<object>.png.
    """

    ensure_output_dirs(config)

    project_root = resolve_project_root(config)
    rows: List[Dict[str, Any]] = []
    env: Optional[UR10EDG5FGraspEnv] = None

    def add_check(name: str, status: str, details: Any = "") -> None:
        """Добавляет результат одной pre-flight проверки в общий отчет."""

        rows.append({
            "object": config.object_shape,
            "check": name,
            "status": status,
            "details": str(details),
        })

    def check_bool(name: str, condition: bool, details: Any = "") -> None:
        """Добавляет проверку со статусом OK/FAIL на основании булева условия."""

        add_check(name, "OK" if bool(condition) else "FAIL", details)

    # ---------------------------------------------------------------------
    # 1. Проверка файловой структуры проекта и исходных XML Tesollo.
    # ---------------------------------------------------------------------
    try:
        tesollo_root = resolve_tesollo_root(config)
        check_bool("project_root_exists", project_root.exists(), project_root)
        check_bool("tesollo_root_exists", tesollo_root.exists(), tesollo_root)

        current_script = Path(__file__).resolve()
        check_bool("main_script_exists", current_script.exists(), current_script.name)

        scene_dg = tesollo_root / "robot" / "scene_dg.xml"
        dg5f_xml = tesollo_root / "robot" / "dg5f_right.xml"
        ur10e_dg5f_xml = tesollo_root / "robot" / "ur10edg5f.xml"

        check_bool("scene_dg_exists", scene_dg.exists(), scene_dg)
        check_bool("dg5f_xml_exists", dg5f_xml.exists(), dg5f_xml)
        check_bool("ur10e_dg5f_xml_exists", ur10e_dg5f_xml.exists(), ur10e_dg5f_xml)
    except Exception as exc:
        add_check("project_file_structure", "FAIL", repr(exc))
        tesollo_root = project_root / config.tesollo_root

    # ---------------------------------------------------------------------
    # 2. Проверка импорта ключевых библиотек.
    # ---------------------------------------------------------------------
    for module_name in [
        "torch",
        "torchvision",
        "gymnasium",
        "mujoco",
        "stable_baselines3",
        "pinocchio",
    ]:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "version unknown")
            add_check(f"import_{module_name}", "OK", version)
        except Exception as exc:
            add_check(f"import_{module_name}", "FAIL", repr(exc))

    # Проверка доступности основных объектов текущего проекта.
    try:
        _ = ExperimentConfig
        _ = UR10EDG5FGraspEnv
        _ = make_grasp_scene_xml
        add_check(
            "import_project_module",
            "OK",
            "ExperimentConfig, UR10EDG5FGraspEnv, make_grasp_scene_xml доступны",
        )
    except Exception as exc:
        add_check("import_project_module", "FAIL", repr(exc))

    try:
        # -----------------------------------------------------------------
        # 3. Генерация сцены и создание среды.
        # -----------------------------------------------------------------
        generated_scene = make_grasp_scene_xml(config)
        check_bool("generated_scene_exists", generated_scene.exists(), generated_scene)

        env = UR10EDG5FGraspEnv(config)
        add_check("env_created", "OK", "Среда UR10e + DG-5F создана")

        check_bool("action_space_exists", hasattr(env, "action_space"), env.action_space)
        check_bool("observation_space_exists", hasattr(env, "observation_space"), env.observation_space)

        obs, info = env.reset(seed=config.seed)
        add_check("reset_environment", "OK", f"obs_shape={np.asarray(obs).shape}")

        test_action = ((env.action_space.low + env.action_space.high) / 2.0).astype(np.float32)
        obs2, reward, terminated, truncated, step_info = env.step(test_action)
        one_step_ok = np.all(np.isfinite(np.asarray(obs2))) and np.isfinite(float(reward))
        check_bool(
            "one_step_works",
            bool(one_step_ok),
            {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "obs_shape": np.asarray(obs2).shape,
            },
        )
        check_bool("env_basic_check", bool(one_step_ok), "reset() и step() выполнены")

        # -----------------------------------------------------------------
        # 4. Проверка расположения объекта.
        # -----------------------------------------------------------------
        obs, info = env.reset(seed=config.seed)
        object_pos, _ = env._get_object_pose()
        ee_pos = env._get_ee_position()
        initial_distance = float(np.linalg.norm(object_pos - ee_pos))
        check_bool(
            "object_not_too_far",
            initial_distance <= 0.35,
            {
                "object_pos": object_pos.tolist(),
                "ee_pos": ee_pos.tolist(),
                "initial_distance": initial_distance,
                "threshold": 0.35,
            },
        )

        # -----------------------------------------------------------------
        # 5. Проверка камер и сохранение диагностических кадров.
        # -----------------------------------------------------------------
        images = env.render()
        camera_ok = isinstance(images, dict) and len(images) > 0
        nonempty_images: Dict[str, Any] = {}

        if camera_ok:
            figures_dir = project_root / config.figures_dir
            for camera_name, image in images.items():
                image_array = np.asarray(image)
                is_nonempty = (
                    image_array.ndim == 3
                    and image_array.shape[0] > 0
                    and image_array.shape[1] > 0
                    and image_array.shape[2] >= 3
                    and float(np.std(image_array)) > 0.0
                )
                nonempty_images[camera_name] = {
                    "shape": image_array.shape,
                    "std": float(np.std(image_array)),
                    "nonempty": bool(is_nonempty),
                }
                if is_nonempty:
                    Image.fromarray(image_array[:, :, :3].astype(np.uint8)).save(
                        figures_dir / f"preflight_{config.object_shape}_{camera_name}.png"
                    )

        check_bool(
            "camera_returns_nonempty_image",
            bool(camera_ok and all(item["nonempty"] for item in nonempty_images.values())),
            nonempty_images if camera_ok else type(images),
        )
        add_check(
            "save_camera_snapshots",
            "OK" if camera_ok else "FAIL",
            str(project_root / config.figures_dir),
        )

        # -----------------------------------------------------------------
        # 6. Проверка action_space и влияния действий на суставы.
        # -----------------------------------------------------------------
        low = np.asarray(env.action_space.low)
        high = np.asarray(env.action_space.high)
        action_shape_ok = low.shape == high.shape == env.action_space.shape
        check_bool(
            "action_space_exists",
            bool(action_shape_ok),
            {
                "shape": env.action_space.shape,
                "low_min": float(low.min()),
                "low_max": float(low.max()),
                "high_min": float(high.min()),
                "high_max": float(high.max()),
            },
        )

        changes: List[float] = []
        rewards_for_actions: List[float] = []
        for action_name, action in [
            ("mid", ((low + high) / 2.0).astype(np.float32)),
            ("low", low.astype(np.float32)),
            ("high", high.astype(np.float32)),
        ]:
            obs, info = env.reset(seed=config.seed)
            q0 = env.data.qpos[env.joints_qpos_idx].copy()
            last_reward = 0.0
            for _ in range(5):
                obs, last_reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            q1 = env.data.qpos[env.joints_qpos_idx].copy()
            delta = float(np.linalg.norm(q1 - q0))
            changes.append(delta)
            rewards_for_actions.append(float(last_reward))

        max_change = max(changes) if changes else 0.0
        check_bool("action_changes_joint_positions", max_change > 1e-5, {"changes": changes, "max_qpos_delta": max_change})
        check_bool("action_not_explosive", max_change < 10.0, {"changes": changes, "max_qpos_delta": max_change})

        # -----------------------------------------------------------------
        # 7. Проверка, что reward не является константным.
        # -----------------------------------------------------------------
        rewards: List[float] = []
        contact_counts: List[float] = []
        bad_contact_counts: List[float] = []
        escaped_flags: List[bool] = []
        hyperextension_values: List[float] = []

        obs, info = env.reset(seed=config.seed)
        for _ in range(min(100, config.max_episode_steps)):
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(random_action)
            rewards.append(float(reward))
            contact_counts.append(float(info.get("contact_count", 0.0)))
            bad_contact_counts.append(float(info.get("bad_contact_count", 0.0)))
            escaped_flags.append(bool(info.get("object_escaped", False)))
            hyperextension_values.append(float(info.get("finger_hyperextension", 0.0)))
            if terminated or truncated:
                obs, info = env.reset(seed=config.seed)

        reward_std = float(np.std(rewards)) if rewards else 0.0
        check_bool(
            "reward_is_not_constant",
            reward_std > 1e-8,
            {
                "reward_min": float(np.min(rewards)) if rewards else None,
                "reward_max": float(np.max(rewards)) if rewards else None,
                "reward_std": reward_std,
            },
        )

        # Сохраняем график reward_trace в figures/.
        try:
            import matplotlib.pyplot as plt

            fig_path = project_root / config.figures_dir / f"preflight_reward_trace_{config.object_shape}.png"
            plt.figure(figsize=(8, 4))
            plt.plot(rewards)
            plt.title(f"Pre-flight reward trace: {config.object_shape}")
            plt.xlabel("Step")
            plt.ylabel("Reward")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            add_check("reward_plot_saved", "OK", fig_path)
        except Exception as exc:
            add_check("reward_plot_saved", "WARN", repr(exc))

        add_check(
            "contact_diagnostics_available",
            "OK",
            {
                "max_contact_count": max(contact_counts) if contact_counts else 0.0,
                "max_bad_contact_count": max(bad_contact_counts) if bad_contact_counts else 0.0,
                "any_object_escaped": any(escaped_flags),
                "max_finger_hyperextension": max(hyperextension_values) if hyperextension_values else 0.0,
            },
        )

        # -----------------------------------------------------------------
        # 8. Проверка контактов scripted-действиями.
        # -----------------------------------------------------------------
        contact_results: List[Tuple[str, float, float, float]] = []
        for action_name, action in [
            ("low", low.astype(np.float32)),
            ("high", high.astype(np.float32)),
            ("mid", ((low + high) / 2.0).astype(np.float32)),
        ]:
            obs, info = env.reset(seed=config.seed)
            max_contact = 0.0
            max_lift = -1e9
            last_reward = 0.0
            for _ in range(min(150, config.max_episode_steps)):
                obs, reward, terminated, truncated, info = env.step(action)
                max_contact = max(max_contact, float(info.get("contact_count", 0.0)))
                max_lift = max(max_lift, float(info.get("object_lift", 0.0)))
                last_reward = float(reward)
                if terminated or truncated:
                    break
            contact_results.append((action_name, max_contact, max_lift, last_reward))

        contact_detected = any(item[1] > 0.0 for item in contact_results)
        add_check(
            "contact_can_be_detected_under_scripted_actions",
            "OK" if contact_detected else "WARN",
            contact_results,
        )

        # -----------------------------------------------------------------
        # 9. Программная проверка success-логики.
        # -----------------------------------------------------------------
        obs, info = env.reset(seed=config.seed)
        original_get_contact_flags = env._get_contact_flags
        original_get_ee_position = env._get_ee_position

        try:
            env._get_contact_flags = lambda: np.array([1, 1, 0, 0, 0], dtype=np.float32)

            current_qpos = env.data.qpos[env.object_qpos_addr:env.object_qpos_addr + 7].copy()
            current_qpos[2] = env.initial_object_z + config.success_height_delta + 0.01
            env.data.qpos[env.object_qpos_addr:env.object_qpos_addr + 7] = current_qpos
            mujoco.mj_forward(env.model, env.data)

            object_pos_for_success, _ = env._get_object_pose()
            env._get_ee_position = lambda: object_pos_for_success.copy()

            neutral_action = ((low + high) / 2.0).astype(np.float32)
            forced_reward, forced_info = env._compute_reward(neutral_action)

            success_logic_ok = bool(forced_info.get("success", False))
            check_bool(
            "success_logic_can_return_true",
            success_logic_ok,
            {"reward": float(forced_reward), "info": forced_info},
            )
        finally:
            env._get_contact_flags = original_get_contact_flags
            env._get_ee_position = original_get_ee_position

    except Exception as exc:
        add_check("preflight_exception", "FAIL", repr(exc))

    finally:
        if env is not None:
            env.close()

    # ---------------------------------------------------------------------
    # 10. Сохранение CSV/TXT отчета.
    # ---------------------------------------------------------------------
    report_dir = project_root / config.preflight_reports_dir
    report_path = report_dir / f"preflight_report_{config.object_shape}.csv"
    txt_path = report_dir / f"preflight_report_{config.object_shape}.txt"

    df = pd.DataFrame(rows)
    df.to_csv(report_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write("Pre-flight диагностика UR10e + DG-5F RL Project\n")
        file.write(f"Object shape: {config.object_shape}\n")
        file.write(f"Project root: {project_root}\n")
        file.write(f"Tesollo root: {project_root / config.tesollo_root}\n\n")
        for row in rows:
            file.write(f"[{row['status']}] {row['check']}: {row['details']}\n")

    append_log(config, f"Сохранен полный pre-flight отчет: {report_path}")
    return report_path

def create_metric_figures(metrics: List[Dict[str, float]], config: ExperimentConfig) -> List[Path]:
    """Строит и сохраняет основные графики по итоговым метрикам в figures/.

    Формируются отдельные изображения для Success Rate, Average Reward, ошибок MAE/RMSE/ITAE и времени обучения.
    """

    ensure_output_dirs(config)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        append_log(config, f"Matplotlib недоступен, графики не сохранены: {exc}")
        return []

    if not metrics:
        return []

    df = pd.DataFrame(metrics)
    figures_dir = resolve_project_root(config) / config.figures_dir
    saved_paths: List[Path] = []

    metric_groups = [
        ("success_rate", "Success Rate", "success_rate.png"),
        ("average_reward", "Average Reward", "average_reward.png"),
        ("training_time_sec", "Training Time, sec", "training_time_sec.png"),
        ("mean_inference_time_sec", "Mean Inference Time, sec", "mean_inference_time_sec.png"),
        ("mae", "MAE", "mae.png"),
        ("rmse", "RMSE", "rmse.png"),
        ("itae", "ITAE", "itae.png"),
    ]

    for metric_name, title, filename in metric_groups:
        if metric_name not in df.columns:
            continue

        plot_df = df.copy()
        plot_df["label"] = plot_df["object"].astype(str) + " / " + plot_df["algorithm"].astype(str)

        plt.figure(figsize=(10, 5))
        plt.bar(plot_df["label"], plot_df[metric_name])
        plt.title(title)
        plt.xlabel("Object / Algorithm")
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        output_path = figures_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()
        saved_paths.append(output_path)

    if {"mae", "mse", "rmse", "itae"}.issubset(df.columns):
        error_df = df[["object", "algorithm", "mae", "mse", "rmse", "itae"]].copy()
        error_df["label"] = error_df["object"].astype(str) + " / " + error_df["algorithm"].astype(str)

        plt.figure(figsize=(11, 6))
        x = np.arange(len(error_df))
        width = 0.2
        for idx, metric_name in enumerate(["mae", "mse", "rmse", "itae"]):
            plt.bar(x + (idx - 1.5) * width, error_df[metric_name], width, label=metric_name.upper())
        plt.title("Error metrics")
        plt.xlabel("Object / Algorithm")
        plt.ylabel("Metric value")
        plt.xticks(x, error_df["label"], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        output_path = figures_dir / "error_metrics_comparison.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        saved_paths.append(output_path)

    append_log(config, f"Сохранены графики метрик: {len(saved_paths)} файлов")
    return saved_paths

# =============================================================================
# Блок 5. Генерация MuJoCo-сцены UR10e + DG-5F + объект
# =============================================================================

def object_geom_xml(config: ExperimentConfig) -> str:
    """Возвращает XML-описание геометрии объекта манипулирования.

    Поддерживаются три формы: куб, цилиндр и сфера.
    """

    rgba = {
        "cube": "0.90 0.10 0.10 1",
        "cylinder": "0.10 0.70 0.10 1",
        "sphere": "0.10 0.25 0.90 1",
    }.get(config.object_shape, "0.90 0.10 0.10 1")

    if config.object_shape == "cube":
        return (
            f'<geom name="{config.object_geom_name}" type="box" '
            f'size="{config.object_size} {config.object_size} {config.object_size}" '
            f'mass="{config.object_mass}" rgba="{rgba}" class="collision_obj"/>'
        )

    if config.object_shape == "cylinder":
        return (
            f'<geom name="{config.object_geom_name}" type="cylinder" '
            f'size="{config.object_size} {2.0 * config.object_size}" '
            f'mass="{config.object_mass}" rgba="{rgba}" class="collision_obj"/>'
        )

    if config.object_shape == "sphere":
        return (
            f'<geom name="{config.object_geom_name}" type="sphere" '
            f'size="{config.object_size}" mass="{config.object_mass}" '
            f'rgba="{rgba}" class="collision_obj"/>'
        )

    raise ValueError(
        "object_shape должен быть одним из значений: cube, cylinder, sphere."
    )

def make_grasp_scene_xml(config: ExperimentConfig) -> Path:
    """Создает MJCF-сцену для задачи захвата UR10e + DG-5F.

    Сцена включает реальную модель Tesollo ur10edg5f.xml, стол, объект манипулирования и две справочные XML-камеры.
    GIF строится отдельными свободными камерами, чтобы можно было регулировать их положение из Python.
    """

    ensure_output_dirs(config)

    project_root = resolve_project_root(config)
    tesollo_root = resolve_tesollo_root(config)
    robot_dir = tesollo_root / "robot"

    if not (robot_dir / "ur10edg5f.xml").exists():
        raise FileNotFoundError(
            f"Не найден файл модели UR10e + DG-5F: {robot_dir / 'ur10edg5f.xml'}"
        )

    scene_dir = project_root / config.generated_scenes_dir
    scene_path = scene_dir / f"scene_ur10e_dg5f_{config.object_shape}.xml"

    object_xml = object_geom_xml(config)

    # include использует абсолютный путь, чтобы сцена работала из любого каталога.
    include_path = (robot_dir / "ur10edg5f.xml").as_posix()

    xml = f"""<mujoco model="ur10e_dg5f_grasp_task">
  <include file="{include_path}"/>

  <compiler meshdir="{(robot_dir / "assets").as_posix()}" angle="radian" autolimits="true"/>

  <option integrator="RK4" timestep="{config.sim_timestep}">
    <flag multiccd="enable" nativeccd="enable"/>
  </option>

  <statistic center="-0.15 -0.55 0.20" extent="0.80"/>

  <default>
    <default class="collision_obj">
      <geom condim="4" contype="5" conaffinity="1" priority="1"
            solref="0.001 1.5" solimp="0.9 0.95 0.001"
            friction="1 1 0.005"/>
    </default>
  </default>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <global azimuth="145" elevation="-24"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
             rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="false"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <body name="table" pos="-0.05 -0.60 0.075">
      <geom name="table_geom" type="box" size="0.55 0.35 0.075"
            rgba="0.8 0.8 0.8 1" class="collision_obj"/>
      <geom name="target_area" type="box" pos="-0.10 0.05 0.076"
            size="0.16 0.16 0.001" rgba="0.6 0.4 0.2 1.0"
            contype="0" conaffinity="0"/>
    </body>

    <body name="{config.object_name}"
          pos="{config.object_initial_pos_x} {config.object_initial_pos_y} {config.object_initial_pos_z}"
          quat="1 0 0 0">
      <freejoint name="{config.object_joint_name}"/>
      {object_xml}
    </body>

    <camera name="cam_front" pos="-0.15 -1.20 0.45"
            quat="0.819134 0.573602 0 0" fovy="45"/>
    <camera name="cam_side" pos="0.45 -0.75 0.45"
            quat="0.729512 0.472006 0.297701 0.395471" fovy="45"/>
  </worldbody>
</mujoco>
"""

    scene_path.write_text(xml, encoding="utf-8")
    return scene_path

# =============================================================================
# Блок 6. Визуальный экстрактор ResNet18
# =============================================================================

class ResNet18FeatureExtractor(nn.Module):
    """Экстрактор визуальных признаков на основе ResNet18.
    
    Классификационная голова удаляется, а выходом является вектор признаков размерности 512.
    При отсутствии локально доступных весов ImageNet сеть создается без предобученных весов, чтобы проект оставался запускаемым.
    """

    def __init__(self, use_pretrained: bool = True, freeze_backbone: bool = True):
        """Инициализирует ResNet18 и удаляет классификационную голову."""

        super().__init__()

        weights = None
        if use_pretrained:
            try:
                weights = ResNet18_Weights.DEFAULT
            except Exception:
                weights = None

        try:
            backbone = resnet18(weights=weights)
        except Exception:
            backbone = resnet18(weights=None)

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.output_dim = 512

        if freeze_backbone:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Преобразует batch RGB-изображений в batch визуальных признаков."""

        features = self.feature_extractor(image_tensor)
        return features.view(features.size(0), -1)


def build_image_transform(image_size: int) -> transforms.Compose:
    """Создает преобразование PIL/NumPy изображения в тензор ResNet18."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

# =============================================================================
# Блок 7. Gymnasium-среда UR10e + DG-5F
# =============================================================================

class UR10EDG5FGraspEnv(gym.Env):
    """Среда захвата с управлением UR10e и DG-5F.

    Action space:
    - 26 значений в диапазоне [-1, 1];
    - первые 6 управляют суставами UR10e;
    - следующие 20 управляют суставами DG-5F.

    Observation space:
    - визуальные признаки ResNet18;
    - положения и скорости суставов;
    - положение end-effector;
    - положение объекта;
    - относительный вектор object - end_effector;
    - признаки контакта пальцев;
    - высота подъема и смещение объекта.
    """

    metadata = {"render_modes": ["rgb_array"]}

    ur_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    dg_joint_names = [
        "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
        "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
        "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
        "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
        "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4",
    ]

    actuator_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
        "rj_dg_1_1_ctrl", "rj_dg_1_2_ctrl", "rj_dg_1_3_ctrl", "rj_dg_1_4_ctrl",
        "rj_dg_2_1_ctrl", "rj_dg_2_2_ctrl", "rj_dg_2_3_ctrl", "rj_dg_2_4_ctrl",
        "rj_dg_3_1_ctrl", "rj_dg_3_2_ctrl", "rj_dg_3_3_ctrl", "rj_dg_3_4_ctrl",
        "rj_dg_4_1_ctrl", "rj_dg_4_2_ctrl", "rj_dg_4_3_ctrl", "rj_dg_4_4_ctrl",
        "rj_dg_5_1_ctrl", "rj_dg_5_2_ctrl", "rj_dg_5_3_ctrl", "rj_dg_5_4_ctrl",
    ]

    finger_prefixes = [f"rl_dg_{idx}_" for idx in range(1, 6)]

    def __init__(self, config: ExperimentConfig):
        """Создает MuJoCo-модель, индексы, ResNet18 и spaces."""

        super().__init__()

        self.config = config
        self.project_root = resolve_project_root(config)

        self.scene_path = make_grasp_scene_xml(config)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = config.sim_timestep
        self.frame_skip = max(1, int(round((1.0 / config.control_hz) / config.sim_timestep)))

        self.step_count = 0
        self.success = False
        self.initial_object_z = config.object_initial_pos_z

        self._setup_indices()
        self._setup_control_ranges()
        self._setup_resnet()
        self._setup_spaces()

        self.renderer = mujoco.Renderer(
            self.model,
            height=config.gif_height,
            width=config.gif_width,
        )

    def _setup_indices(self) -> None:
        """Находит индексы суставов, актуаторов, объекта и end-effector."""

        self.joint_names = self.ur_joint_names + self.dg_joint_names

        self.joints_qpos_idx: List[int] = []
        self.joints_qvel_idx: List[int] = []

        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise RuntimeError(f"В модели не найден сустав: {joint_name}")

            self.joints_qpos_idx.append(int(self.model.jnt_qposadr[joint_id]))
            self.joints_qvel_idx.append(int(self.model.jnt_dofadr[joint_id]))

        self.joints_qpos_idx = np.array(self.joints_qpos_idx, dtype=np.int64)
        self.joints_qvel_idx = np.array(self.joints_qvel_idx, dtype=np.int64)

        self.actuator_idx: List[int] = []
        for actuator_name in self.actuator_names:
            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            if actuator_id < 0:
                raise RuntimeError(f"В модели не найден актуатор: {actuator_name}")

            self.actuator_idx.append(int(actuator_id))

        self.actuator_idx = np.array(self.actuator_idx, dtype=np.int64)

        self.object_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.config.object_name,
        )
        if self.object_body_id < 0:
            raise RuntimeError(f"В модели не найден объект: {self.config.object_name}")

        self.object_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            self.config.object_joint_name,
        )
        if self.object_joint_id < 0:
            raise RuntimeError(
                f"В модели не найден freejoint объекта: {self.config.object_joint_name}"
            )

        self.object_qpos_addr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_qvel_addr = int(self.model.jnt_dofadr[self.object_joint_id])

        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "rl_dg_mount",
        )
        if self.ee_body_id < 0:
            self.ee_body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "rl_dg_palm",
            )

        if self.ee_body_id < 0:
            raise RuntimeError("Не найден body end-effector: rl_dg_mount/rl_dg_palm.")

    def _setup_control_ranges(self) -> None:
        """Создает массивы диапазонов управления для актуаторов."""

        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_idx, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_idx, 1].copy()

        # Если autolimits не задал диапазоны, используем безопасный запас.
        invalid = self.ctrl_high <= self.ctrl_low
        self.ctrl_low[invalid] = -10.0
        self.ctrl_high[invalid] = 10.0

        self.last_ctrl = np.zeros(len(self.actuator_idx), dtype=np.float64)

    def _setup_resnet(self) -> None:
        """Создает ResNet18 и preprocessing изображения."""

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() and self.config.device == "auto"
            else ("cuda" if torch.cuda.is_available() and self.config.device == "auto" else "cpu")
        )

        self.resnet = ResNet18FeatureExtractor(
            use_pretrained=self.config.use_pretrained_resnet,
            freeze_backbone=self.config.freeze_resnet,
        ).to(self.device)

        self.resnet.eval()
        self.image_transform = build_image_transform(self.config.image_size)

    def _setup_spaces(self) -> None:
        """Создает action_space и observation_space для Stable-Baselines3."""

        self.action_dim = len(self.actuator_idx)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # 512 visual + qpos 26 + qvel 26 + ee 3 + obj 3 + rel 3 + contacts 5 + lift 1 + xy_shift 1
        obs_dim = 512 + 26 + 26 + 3 + 3 + 3 + 5 + 1 + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _initial_joint_pose(self) -> np.ndarray:
        """Возвращает начальную позу UR10e + DG-5F."""

        ur = np.array(
            [
                self.config.initial_ur_qpos_0,
                self.config.initial_ur_qpos_1,
                self.config.initial_ur_qpos_2,
                self.config.initial_ur_qpos_3,
                self.config.initial_ur_qpos_4,
                self.config.initial_ur_qpos_5,
            ],
            dtype=np.float64,
        )

        fingers = np.zeros(20, dtype=np.float64)
        return np.concatenate([ur, fingers])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Сбрасывает MuJoCo-сцену, объект и управляющие сигналы."""

        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        initial_pose = self._initial_joint_pose()
        for idx, qpos_addr in enumerate(self.joints_qpos_idx):
            self.data.qpos[qpos_addr] = initial_pose[idx]
            self.data.qvel[self.joints_qvel_idx[idx]] = 0.0

        self.last_ctrl = np.clip(initial_pose.copy(), self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = self.last_ctrl

        dx = np.random.uniform(-self.config.object_random_xy, self.config.object_random_xy)
        dy = np.random.uniform(-self.config.object_random_xy, self.config.object_random_xy)

        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3] = np.array(
            [
                self.config.object_initial_pos_x + dx,
                self.config.object_initial_pos_y + dy,
                self.config.object_initial_pos_z,
            ],
            dtype=np.float64,
        )
        self.data.qpos[self.object_qpos_addr + 3:self.object_qpos_addr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.data.qvel[self.object_qvel_addr:self.object_qvel_addr + 6] = 0.0

        self.initial_object_z = float(self.data.qpos[self.object_qpos_addr + 2])
        self.step_count = 0
        self.success = False
        self.object_has_been_lifted = False

        mujoco.mj_forward(self.model, self.data)

        # Базовые значения для progress-based reward.
        # Reach Reward должен оценивать именно изменение расстояния между шагами,
        # поэтому после reset фиксируем стартовую ошибку слежения.
        object_pos, _ = self._get_object_pose()
        ee_pos = self._get_ee_position()
        self.previous_tracking_error = float(np.linalg.norm(object_pos - ee_pos))
        self.previous_object_lift = 0.0

        return self._get_observation(), self._get_info_stub()

    def step(self, action: np.ndarray):
        """Применяет действие UR10e + DG-5F, делает шаг физики и считает reward.

        Сначала действие агента ограничивается общим диапазоном [-1, 1].
        Затем дополнительно ограничиваются компоненты, относящиеся к пальцам DG-5F,
        чтобы предотвращать сильное переразгибание пальцев в направлении, противоположном естественному закрытию захвата.
        """

        raw_action = np.asarray(action, dtype=np.float64)
        raw_action = np.clip(raw_action, -1.0, 1.0)

        action, action_safety_info = self._sanitize_action(raw_action)

        old_qpos = self.data.qpos.copy()
        old_qvel = self.data.qvel.copy()
        old_ctrl = self.data.ctrl.copy()
        old_last_ctrl = self.last_ctrl.copy()

        self._apply_action(action)

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        ee_outside_table = self._is_ee_outside_table()

        if ee_outside_table:
            self.data.qpos[:] = old_qpos
            self.data.qvel[:] = old_qvel
            self.data.ctrl[:] = old_ctrl
            self.last_ctrl = old_last_ctrl.copy()
            mujoco.mj_forward(self.model, self.data)

        reward, info = self._compute_reward(action)
        info["ee_outside_table"] = bool(ee_outside_table)
        info.update(action_safety_info)

        self.step_count += 1

        terminated = bool(info["success"] or info["object_escaped"])
        truncated = self.step_count >= self.config.max_episode_steps

        observation = self._get_observation()

        return observation, reward, terminated, truncated, info

    def _sanitize_action(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ограничивает действие агента перед передачей в MuJoCo.

        Первые 6 компонент управляют UR10e и сохраняют диапазон [-1, 1].
        Последние 20 компонент управляют пальцами DG-5F и дополнительно ограничиваются диапазоном [finger_action_min, finger_action_max].
        Это не меняет размерность action space, но предотвращает устойчивые попытки агента выгибать пальцы в нерабочую сторону.
        """

        safe_action = action.copy()

        ur_action = safe_action[:6]
        finger_action_before = safe_action[6:].copy()
        finger_action_after = np.clip(
            finger_action_before,
            self.config.finger_action_min,
            self.config.finger_action_max,
        )

        safe_action[:6] = ur_action
        safe_action[6:] = finger_action_after

        clipped_finger_action = np.abs(finger_action_after - finger_action_before)
        clipped_finger_action_sum = float(np.sum(clipped_finger_action))

        return safe_action, {
            "finger_action_min_before_clip": float(np.min(finger_action_before)) if finger_action_before.size else 0.0,
            "finger_action_max_before_clip": float(np.max(finger_action_before)) if finger_action_before.size else 0.0,
            "finger_action_min_after_clip": float(np.min(finger_action_after)) if finger_action_after.size else 0.0,
            "finger_action_max_after_clip": float(np.max(finger_action_after)) if finger_action_after.size else 0.0,
            "finger_action_clip_sum": clipped_finger_action_sum,
        }

    def _apply_action(self, action: np.ndarray) -> None:
        """Преобразует нормированные действия в целевые позиции актуаторов.

        Первые 6 действий управляют UR10e, следующие 20 — DG-5F.
        Действия интерпретируются как приращения целевых положений, что делает управление более устойчивым, чем абсолютная установка углов.
        """

        ur_delta = action[:6] * self.config.ur_action_delta
        dg_delta = action[6:] * self.config.finger_action_delta

        delta = np.concatenate([ur_delta, dg_delta])
        target_ctrl = self.last_ctrl + delta
        target_ctrl = np.clip(target_ctrl, self.ctrl_low, self.ctrl_high)

        self.last_ctrl = target_ctrl.copy()
        self.data.ctrl[:] = target_ctrl

    def _get_object_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает положение и кватернион объекта манипулирования."""

        pos = self.data.xpos[self.object_body_id].copy()
        quat = self.data.xquat[self.object_body_id].copy()
        return pos, quat

    def _get_ee_position(self) -> np.ndarray:
        """Возвращает положение mount/palm DG-5F как приближение end-effector."""

        return self.data.xpos[self.ee_body_id].copy()

    def _is_ee_outside_table(self) -> bool:
        """Проверяет, вышел ли end-effector за габариты стола более чем на table_margin."""

        ee_pos = self._get_ee_position()
        x_min = self.config.table_x_min - self.config.table_margin
        x_max = self.config.table_x_max + self.config.table_margin
        y_min = self.config.table_y_min - self.config.table_margin
        y_max = self.config.table_y_max + self.config.table_margin

        return bool(
            ee_pos[0] < x_min
            or ee_pos[0] > x_max
            or ee_pos[1] < y_min
            or ee_pos[1] > y_max
        )

    def _compute_tracking_error(self) -> float:
        """Вычисляет расстояние между end-effector и объектом."""

        object_pos, _ = self._get_object_pose()
        ee_pos = self._get_ee_position()
        return float(np.linalg.norm(object_pos - ee_pos))

    def _get_contact_flags(self) -> np.ndarray:
        """Определяет контакт каждого пальца DG-5F с объектом.

        MuJoCo хранит контакты как пары geom, поэтому проверка выполняется по именам geom.
        """

        flags = np.zeros(5, dtype=np.float32)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(contact.geom1),
            ) or ""

            geom2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(contact.geom2),
            ) or ""

            pair = f"{geom1} {geom2}"

            if self.config.object_name not in pair and self.config.object_geom_name not in pair:
                continue

            for finger_idx, prefix in enumerate(self.finger_prefixes):
                if prefix in pair:
                    flags[finger_idx] = 1.0

        return flags

    def _get_object_contact_pair_names(self) -> List[Tuple[str, str]]:
        """Возвращает имена geom, которые контактируют с объектом манипулирования.

        MuJoCo хранит контакты на уровне geom, поэтому для диагностики reward используется именно список пар geom.
        Функция отбирает только те контакты, где один из geom относится к объекту захвата.
        """

        contact_pairs: List[Tuple[str, str]] = []

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(contact.geom1),
            ) or ""

            geom2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(contact.geom2),
            ) or ""

            pair = f"{geom1} {geom2}"

            if self.config.object_name in pair or self.config.object_geom_name in pair:
                contact_pairs.append((geom1, geom2))

        return contact_pairs

    def _is_finger_geom(self, geom_name: str) -> bool:
        """Проверяет, относится ли geom к рабочей части одного из пальцев DG-5F."""

        return any(prefix in geom_name for prefix in self.finger_prefixes)

    def _is_ignored_environment_geom(self, geom_name: str) -> bool:
        """Проверяет, относится ли geom к окружению, а не к роботу.

        Контакт объекта со столом или полом не должен считаться вредным контактом с роботом,
        иначе агент будет штрафоваться уже в начале эпизода, когда объект естественно лежит на поверхности.
        """

        ignored_tokens = (
            "floor",
            "ground",
            "table",
            "plane",
            "world",
        )

        return any(token in geom_name.lower() for token in ignored_tokens)

    def _get_bad_contact_info(self) -> Tuple[bool, int, List[str]]:
        """Определяет вредные контакты объекта с нерабочими частями робота.

        Полезным считается контакт объекта с пальцами DG-5F.
        Вредным считается контакт объекта с ладонью, тыльной стороной, основанием кисти, фланцем, звеньями UR10e или другими нерабочими частями манипулятора.
        Контакты объекта со столом, полом и самим объектом игнорируются.
        """

        bad_geoms: List[str] = []

        for geom1, geom2 in self._get_object_contact_pair_names():
            if self.config.object_name in geom1 or self.config.object_geom_name in geom1:
                other_geom = geom2
            else:
                other_geom = geom1

            other_geom_lower = other_geom.lower()

            if not other_geom:
                continue

            if self.config.object_name in other_geom or self.config.object_geom_name in other_geom:
                continue

            if self._is_ignored_environment_geom(other_geom):
                continue

            if self._is_finger_geom(other_geom):
                continue

            bad_tokens = (
                "palm",
                "base",
                "mount",
                "wrist",
                "flange",
                "ee",
                "tool",
                "ur",
                "shoulder",
                "upper",
                "forearm",
                "elbow",
            )

            if any(token in other_geom_lower for token in bad_tokens):
                bad_geoms.append(other_geom)
            else:
                # Если geom не является пальцем и не является окружением, то для безопасной постановки reward считаем его нерабочим контактом.
                bad_geoms.append(other_geom)

        unique_bad_geoms = sorted(set(bad_geoms))
        return bool(unique_bad_geoms), len(unique_bad_geoms), unique_bad_geoms

    def _has_palm_contact(self) -> bool:
        """Проверяет, контактирует ли объект с ладонью/тыльной частью кисти."""

        bad_contact, _bad_count, bad_geoms = self._get_bad_contact_info()

        if not bad_contact:
            return False

        return any(
            "palm" in geom.lower() or "base" in geom.lower() or "mount" in geom.lower()
            for geom in bad_geoms
        )

    def _compute_finger_hyperextension(self) -> Tuple[float, np.ndarray]:
        """Вычисляет величину переразгибания суставов пальцев DG-5F.

        Используются только 20 суставов пальцев, которые идут после 6 суставов
        UR10e в общем списке joint_names. Если положение сустава ниже
        finger_safe_lower_limit, превышение считается переразгибанием и далее
        штрафуется в reward.
        """

        finger_qpos_idx = self.joints_qpos_idx[6:]
        finger_qpos = self.data.qpos[finger_qpos_idx].copy()

        hyperextension = np.maximum(
            0.0,
            self.config.finger_safe_lower_limit - finger_qpos,
        )

        hyperextension_value = float(np.sum(hyperextension))

        return hyperextension_value, finger_qpos

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Вычисляет reward и диагностическую информацию.

        Reward поощряет приближение end-effector к объекту, контакт объекта именно с пальцами DG-5F и подъем объекта.
        Дополнительно штрафуются чрезмерные действия, высокие скорости суставов, выталкивание объекта из рабочей зоны и контакты объекта с нерабочими частями кисти или UR10e.
        Вредный контакт штрафуется только тогда, когда он не сопровождается полезным контактом пальцев и положительным подъемом объекта.
        Это предотвращает стратегию, при которой агент толкает объект тыльной стороной ладони или звеньями манипулятора,
        но не мешает реальному захвату, если объект уже удерживается пальцами и начинает подниматься.
        """

        object_pos, _ = self._get_object_pose()
        ee_pos = self._get_ee_position()

        contact_flags = self._get_contact_flags()
        contact_count = float(np.sum(contact_flags))

        tracking_error = float(np.linalg.norm(object_pos - ee_pos))
        object_lift = float(object_pos[2] - self.initial_object_z)

        object_initial_xy = np.array(
            [
                self.config.object_initial_pos_x,
                self.config.object_initial_pos_y,
            ],
            dtype=np.float64,
        )

        object_xy_shift = float(np.linalg.norm(object_pos[:2] - object_initial_xy))
        object_escaped = object_xy_shift > self.config.workspace_radius

        bad_contact, bad_contact_count, bad_contact_geoms = self._get_bad_contact_info()
        finger_hyperextension, finger_qpos = self._compute_finger_hyperextension()

        palm_contact = any(
            "palm" in geom.lower() or "base" in geom.lower() or "mount" in geom.lower()
            for geom in bad_contact_geoms
        )

        useful_grasp_progress = (
            contact_count >= self.config.bad_contact_finger_threshold
            and object_lift >= self.config.bad_contact_lift_threshold
        )

        penalize_bad_contact = bool(
            bad_contact
            and not useful_grasp_progress
        )

        success = (
            object_lift >= self.config.success_height_delta
            and contact_count >= self.config.min_success_fingers
            and tracking_error <= self.config.success_distance_threshold
            and not object_escaped
        )

        self.success = bool(success)

        reward = 0.0

        # ------------------------------------------------------------------
        # 1. Reach Reward: награда за уменьшение расстояния до объекта.
        # Важно: агент получает reward не за сам факт нахождения рядом,
        # а за прогресс относительно предыдущего шага.
        # ------------------------------------------------------------------
        previous_tracking_error = float(
            getattr(self, "previous_tracking_error", tracking_error)
        )
        tracking_error_delta = previous_tracking_error - tracking_error
        reach_reward_value = self.config.reach_progress_weight * tracking_error_delta
        reward += reach_reward_value

        # ------------------------------------------------------------------
        # 2. Multi-Finger Contact Reward: поощряется контакт несколькими пальцами.
        # Одиночное касание не считается полноценным хватом и не получает
        # основной контактный бонус. При двух пальцах бонус = contact_reward,
        # при трех = 2 * contact_reward и т.д.
        # ------------------------------------------------------------------
        multi_finger_contact_count = max(0.0, contact_count - 1.0)
        multi_finger_contact_reward_value = (
            self.config.contact_reward * multi_finger_contact_count
        )
        reward += multi_finger_contact_reward_value

        # ------------------------------------------------------------------
        # 3. Lift Reward: поощрение за фактический подъем объекта.
        # ------------------------------------------------------------------
        lift_reward_value = self.config.lift_reward * max(0.0, object_lift)
        reward += lift_reward_value

        # ------------------------------------------------------------------
        # 4. Hold Reward: небольшой бонус за каждый шаг удержания поднятого
        # объекта при контакте минимум двумя пальцами.
        # ------------------------------------------------------------------
        hold_reward_value = 0.0
        if (
            object_lift >= self.config.success_height_delta
            and contact_count >= self.config.min_success_fingers
            and not object_escaped
        ):
            hold_reward_value = self.config.hold_reward
            reward += hold_reward_value

        # ------------------------------------------------------------------
        # 5. Stable Grasp Reward: бонус за устойчивый промежуточный хват,
        # когда есть контакт минимум двумя пальцами и объект уже начал подниматься.
        # ------------------------------------------------------------------
        stable_grasp_reward_value = 0.0
        stable_grasp = bool(
            contact_count >= self.config.min_success_fingers
            and object_lift >= self.config.bad_contact_lift_threshold
            and not object_escaped
        )
        if stable_grasp:
            stable_grasp_reward_value = self.config.stable_grasp_reward
            reward += stable_grasp_reward_value

        # ------------------------------------------------------------------
        # 6. Time Penalty: небольшой штраф за каждый шаг эпизода.
        # ------------------------------------------------------------------
        time_penalty_value = self.config.time_penalty
        reward -= time_penalty_value

        # ------------------------------------------------------------------
        # 7. Drop Penalty: штраф за потерю объекта после того, как он уже был поднят.
        # ------------------------------------------------------------------
        drop_penalty_value = 0.0
        object_was_lifted_before = bool(
            getattr(self, "object_has_been_lifted", False)
        )
        if object_lift >= self.config.drop_lift_threshold:
            self.object_has_been_lifted = True

        object_dropped = bool(
            object_was_lifted_before
            and object_lift <= self.config.drop_height_threshold
        )
        if object_dropped:
            drop_penalty_value = self.config.drop_penalty
            reward -= drop_penalty_value

        # ------------------------------------------------------------------
        # 8. Базовые инженерные штрафы за резкие действия и скорости.
        # ------------------------------------------------------------------
        action_penalty_value = self.config.action_penalty_weight * float(np.linalg.norm(action))
        velocity_penalty_value = self.config.velocity_penalty_weight * float(
            np.linalg.norm(self.data.qvel[self.joints_qvel_idx])
        )
        reward -= action_penalty_value
        reward -= velocity_penalty_value

        if object_escaped:
            reward -= self.config.object_escape_penalty

        if penalize_bad_contact:
            reward -= self.config.bad_contact_penalty * float(bad_contact_count)

        if palm_contact and not useful_grasp_progress:
            reward -= self.config.palm_contact_penalty

        if finger_hyperextension > 0.0:
            reward -= self.config.finger_hyperextension_penalty * finger_hyperextension

        # ------------------------------------------------------------------
        # 9. Success Reward: крупный бонус за выполнение полного критерия успеха.
        # ------------------------------------------------------------------
        if success:
            reward += self.config.success_reward

        self.previous_tracking_error = tracking_error
        self.previous_object_lift = object_lift

        info = {
            "success": bool(success),
            "tracking_error": tracking_error,
            "contact_count": contact_count,
            "contact": bool(contact_count > 0),
            "contact_flags": contact_flags.copy(),
            "object_lift": object_lift,
            "object_z": float(object_pos[2]),
            "object_xy_shift": object_xy_shift,
            "object_escaped": bool(object_escaped),
            "bad_contact": bool(bad_contact),
            "bad_contact_count": float(bad_contact_count),
            "bad_contact_geoms": bad_contact_geoms,
            "penalize_bad_contact": bool(penalize_bad_contact),
            "palm_contact": bool(palm_contact),
            "useful_grasp_progress": bool(useful_grasp_progress),
            "finger_hyperextension": float(finger_hyperextension),
            "finger_hyperextension_penalty": float(
                self.config.finger_hyperextension_penalty * finger_hyperextension
            ),
            "finger_qpos_min": float(np.min(finger_qpos)) if finger_qpos.size else 0.0,
            "finger_qpos_max": float(np.max(finger_qpos)) if finger_qpos.size else 0.0,
            "previous_tracking_error": float(previous_tracking_error),
            "tracking_error_delta": float(tracking_error_delta),
            "reach_reward": float(reach_reward_value),
            "multi_finger_contact_count": float(multi_finger_contact_count),
            "multi_finger_contact_reward": float(multi_finger_contact_reward_value),
            "lift_reward": float(lift_reward_value),
            "hold_reward": float(hold_reward_value),
            "stable_grasp": bool(stable_grasp),
            "stable_grasp_reward": float(stable_grasp_reward_value),
            "time_penalty": float(time_penalty_value),
            "object_was_lifted_before": bool(object_was_lifted_before),
            "object_dropped": bool(object_dropped),
            "drop_penalty": float(drop_penalty_value),
            "action_penalty": float(action_penalty_value),
            "velocity_penalty": float(velocity_penalty_value),
            "ee_pos": ee_pos.copy(),
            "object_pos": object_pos.copy(),
        }

        return float(reward), info

    def _render_free_camera(
        self,
        lookat_x: float,
        lookat_y: float,
        azimuth: float,
        elevation: float,
    ) -> np.ndarray:
        """Рендерит кадр со свободной камеры MuJoCo.

        Обе камеры используют одинаковые camera_distance и camera_height, чтобы визуализация с двух ракурсов была сопоставимой.
        """

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = np.array(
            [
                lookat_x,
                lookat_y,
                self.config.camera_height,
            ],
            dtype=np.float64,
        )
        camera.distance = self.config.camera_distance
        camera.azimuth = azimuth
        camera.elevation = elevation

        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def render_two_cameras(self) -> Dict[str, np.ndarray]:
        """Возвращает кадры с двух регулируемых камер.

        camera_1 — боковой обзор;
        camera_2 — основной обзор.
        """

        frame_1 = self._render_free_camera(
            lookat_x=self.config.camera_1_lookat_x,
            lookat_y=self.config.camera_1_lookat_y,
            azimuth=self.config.camera_1_azimuth,
            elevation=self.config.camera_1_elevation,
        )

        frame_2 = self._render_free_camera(
            lookat_x=self.config.camera_2_lookat_x,
            lookat_y=self.config.camera_2_lookat_y,
            azimuth=self.config.camera_2_azimuth,
            elevation=self.config.camera_2_elevation,
        )

        return {
            self.config.camera_1_name: frame_1,
            self.config.camera_2_name: frame_2,
        }

    def render(self):
        """Возвращает словарь изображений с двух камер."""

        return self.render_two_cameras()

    def _extract_visual_features(self) -> np.ndarray:
        """Извлекает признаки ResNet18 из первой камеры."""

        image = self.render_two_cameras()[self.config.camera_1_name]

        pil_image = Image.fromarray(image)
        tensor = self.image_transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.resnet(tensor)

        return features.cpu().numpy().reshape(-1).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        """Формирует полный вектор состояния для RL-агента."""

        visual_features = self._extract_visual_features()

        joint_pos = self.data.qpos[self.joints_qpos_idx].astype(np.float32)
        joint_vel = self.data.qvel[self.joints_qvel_idx].astype(np.float32)

        object_pos, _ = self._get_object_pose()
        ee_pos = self._get_ee_position()
        relative = object_pos - ee_pos

        contact_flags = self._get_contact_flags()

        object_lift = np.array(
            [float(object_pos[2] - self.initial_object_z)],
            dtype=np.float32,
        )

        object_initial_xy = np.array(
            [
                self.config.object_initial_pos_x,
                self.config.object_initial_pos_y,
            ],
            dtype=np.float32,
        )
        object_xy_shift = np.array(
            [float(np.linalg.norm(object_pos[:2] - object_initial_xy))],
            dtype=np.float32,
        )

        observation = np.concatenate(
            [
                visual_features,
                joint_pos,
                joint_vel,
                ee_pos.astype(np.float32),
                object_pos.astype(np.float32),
                relative.astype(np.float32),
                contact_flags.astype(np.float32),
                object_lift,
                object_xy_shift,
            ]
        )

        return observation.astype(np.float32)

    def _get_info_stub(self) -> Dict[str, Any]:
        """Возвращает базовый info при reset."""

        object_pos, _ = self._get_object_pose()
        ee_pos = self._get_ee_position()
        return {
            "success": False,
            "tracking_error": float(np.linalg.norm(object_pos - ee_pos)),
            "contact_count": 0.0,
            "contact": False,
            "object_lift": 0.0,
            "object_escaped": False,
        }

    def close(self) -> None:
        """Освобождает MuJoCo renderer."""

        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None

# =============================================================================
# Блок 8. Создание моделей SAC/PPO
# =============================================================================

def make_env(config: ExperimentConfig):
    """Создает фабрику среды для DummyVecEnv."""

    def _factory():
        return UR10EDG5FGraspEnv(config)

    return _factory


def make_vec_env(config: ExperimentConfig) -> DummyVecEnv:
    """Создает векторизованную среду Stable-Baselines3."""

    return DummyVecEnv([make_env(config)])


def policy_kwargs(config: ExperimentConfig) -> Dict[str, Any]:
    """Возвращает архитектуру MLP Actor/Critic для SB3."""

    return {
        "net_arch": {
            "pi": [
                config.actor_hidden_1,
                config.actor_hidden_2,
                config.actor_hidden_3,
            ],
            "qf": [
                config.critic_hidden_1,
                config.critic_hidden_2,
                config.critic_hidden_3,
            ],
        }
    }


def create_sac(config: ExperimentConfig, env: DummyVecEnv) -> SAC:
    """Создает SAC-модель для непрерывного управления UR10e + DG-5F."""

    return SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        gamma=config.gamma,
        tau=config.tau,
        policy_kwargs=policy_kwargs(config),
        verbose=1,
        seed=config.seed,
        device=config.device,
    )


def create_ppo(config: ExperimentConfig, env: DummyVecEnv) -> PPO:
    """Создает PPO-модель для сравнения с SAC."""

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        batch_size=min(config.batch_size, 64),
        n_steps=1024,
        gamma=config.gamma,
        policy_kwargs={
            "net_arch": {
                "pi": [
                    config.actor_hidden_1,
                    config.actor_hidden_2,
                    config.actor_hidden_3,
                ],
                "vf": [
                    config.critic_hidden_1,
                    config.critic_hidden_2,
                    config.critic_hidden_3,
                ],
            }
        },
        verbose=1,
        seed=config.seed,
        device=config.device,
    )

def _default_checkpoint_path(
    config: ExperimentConfig,
    algorithm_name: str,
) -> Path:
    """Возвращает стандартный путь к checkpoint модели для указанного алгоритма."""

    project_root = resolve_project_root(config)
    algorithm = algorithm_name.lower()
    return project_root / config.checkpoints_dir / f"{algorithm}_ur10e_dg5f_{config.object_shape}.zip"


def _resolve_resume_checkpoint(
    config: ExperimentConfig,
    algorithm_name: str,
) -> Path:
    """Возвращает путь к checkpoint для продолжения обучения.

    Если путь задан в конфигурации явно, используется он. Если не задан,
    используется стандартный путь сохранения модели для текущего object_shape.
    """

    explicit_path = (
        config.resume_sac_checkpoint
        if algorithm_name.upper() == "SAC"
        else config.resume_ppo_checkpoint
    )

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = resolve_project_root(config) / path
        return path

    return _default_checkpoint_path(config, algorithm_name)


def _save_model_path(
    config: ExperimentConfig,
    algorithm_name: str,
    resumed: bool,
) -> Path:
    """Формирует путь для сохранения новой или дообученной модели без расширения .zip."""

    project_root = resolve_project_root(config)
    algorithm = algorithm_name.lower()
    suffix = ""

    if resumed and config.resumed_model_suffix:
        suffix = f"_{config.resumed_model_suffix}"

    return project_root / config.checkpoints_dir / f"{algorithm}_ur10e_dg5f_{config.object_shape}{suffix}"


def load_or_create_sac(config: ExperimentConfig, env: DummyVecEnv) -> Tuple[SAC, bool, Optional[Path]]:
    """Загружает SAC для продолжения обучения или создает новую модель.

    Возвращает модель, признак загрузки из checkpoint и путь к загруженному checkpoint.
    """

    if config.resume_training:
        checkpoint_path = _resolve_resume_checkpoint(config, "SAC")
        if checkpoint_path.exists():
            print(f"Продолжение обучения SAC из checkpoint: {checkpoint_path}")
            model = SAC.load(
                str(checkpoint_path),
                env=env,
                device=config.device,
            )
            return model, True, checkpoint_path

        print(f"SAC checkpoint не найден, обучение начнется с нуля: {checkpoint_path}")

    return create_sac(config, env), False, None


def load_or_create_ppo(config: ExperimentConfig, env: DummyVecEnv) -> Tuple[PPO, bool, Optional[Path]]:
    """Загружает PPO для продолжения обучения или создает новую модель.

    Возвращает модель, признак загрузки из checkpoint и путь к загруженному checkpoint.
    """

    if config.resume_training:
        checkpoint_path = _resolve_resume_checkpoint(config, "PPO")
        if checkpoint_path.exists():
            print(f"Продолжение обучения PPO из checkpoint: {checkpoint_path}")
            model = PPO.load(
                str(checkpoint_path),
                env=env,
                device=config.device,
            )
            return model, True, checkpoint_path

        print(f"PPO checkpoint не найден, обучение начнется с нуля: {checkpoint_path}")

    return create_ppo(config, env), False, None


# =============================================================================
# Блок 9. Метрики и оценка
# =============================================================================

def compute_error_metrics(
    tracking_errors: List[float],
    dt: float,
) -> Dict[str, float]:
    """Считает MAE, MSE, RMSE и ITAE по ошибке сопровождения объекта."""

    if not tracking_errors:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "itae": float("nan"),
        }

    errors = np.asarray(tracking_errors, dtype=np.float64)
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    times = np.arange(1, len(errors) + 1, dtype=np.float64) * dt
    itae = float(np.sum(times * abs_errors) * dt)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "itae": itae,
    }


def evaluate_policy_custom(
    model,
    config: ExperimentConfig,
    algorithm_name: str,
    object_shape: str,
) -> Dict[str, float]:
    """Оценивает обученную политику и возвращает набор метрик."""

    eval_config = ExperimentConfig(**asdict(config))
    eval_config.object_shape = object_shape

    env = UR10EDG5FGraspEnv(eval_config)

    successes = 0
    rewards: List[float] = []
    errors: List[float] = []
    inference_times: List[float] = []

    try:
        for episode in range(eval_config.eval_episodes):
            obs, _ = env.reset(seed=eval_config.seed + episode)
            done = False
            episode_reward = 0.0

            while not done:
                start_time = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                inference_times.append(time.perf_counter() - start_time)

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                errors.append(float(info["tracking_error"]))

                done = terminated or truncated

            rewards.append(episode_reward)

            if info.get("success", False):
                successes += 1

    finally:
        env.close()

    error_metrics = compute_error_metrics(
        tracking_errors=errors,
        dt=1.0 / eval_config.control_hz,
    )

    result = {
        "object": object_shape,
        "algorithm": algorithm_name,
        "success_rate": successes / max(1, eval_config.eval_episodes),
        "average_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "mean_inference_time_sec": float(np.mean(inference_times)) if inference_times else float("nan"),
    }

    result.update(error_metrics)
    return result

# =============================================================================
# Блок 10. GIF с двух камер
# =============================================================================

def combine_two_camera_frames(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
) -> np.ndarray:
    """Объединяет два RGB-кадра в один горизонтальный кадр."""

    target_height = min(left_frame.shape[0], right_frame.shape[0])

    left = left_frame[:target_height]
    right = right_frame[:target_height]

    return np.concatenate([left, right], axis=1)

def save_rollout_gif(
    model,
    config: ExperimentConfig,
    algorithm_name: str,
    object_shape: str,
) -> Optional[Path]:
    """Сохраняет GIF поведения агента одновременно с двух камер.

    Левая половина GIF — камера 1.
    Правая половина GIF — камера 2, направленная на ладонь/пальцы.
    У обеих камер одинаковые distance и camera_height; регулируются только azimuth/elevation/lookat_x/lookat_y.
    """

    try:
        import imageio.v2 as imageio
    except Exception:
        print("Пакет imageio не установлен, GIF не будет сохранен.")
        return None

    gif_config = ExperimentConfig(**asdict(config))
    gif_config.object_shape = object_shape
    gif_config.render_mode = "rgb_array"

    env = UR10EDG5FGraspEnv(gif_config)
    frames: List[np.ndarray] = []

    try:
        obs, _ = env.reset(seed=gif_config.seed)
        done = False

        while not done and len(frames) < gif_config.max_episode_steps:
            images = env.render()

            frame_1 = images[gif_config.camera_1_name]
            frame_2 = images[gif_config.camera_2_name]

            combined = combine_two_camera_frames(frame_1, frame_2)
            frames.append(combined)

            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

        if not frames:
            return None

        videos_dir = resolve_project_root(gif_config) / gif_config.videos_dir
        videos_dir.mkdir(parents=True, exist_ok=True)

        output_path = videos_dir / (
            f"rollout_{algorithm_name.lower()}_{object_shape}_two_cameras.gif"
        )

        imageio.mimsave(output_path, frames, fps=gif_config.gif_fps)
        return output_path

    except Exception as exc:
        print(f"Не удалось сохранить GIF для {algorithm_name}/{object_shape}: {exc}")
        return None

    finally:
        env.close()

# =============================================================================
# Блок 11. Обучение и сравнение SAC/PPO
# =============================================================================

def train_and_evaluate(config: ExperimentConfig) -> List[Dict[str, float]]:
    """Обучает SAC и PPO на одном объекте и возвращает метрики."""

    ensure_output_dirs(config)
    save_config_snapshot(
        config,
        object_shapes=[config.object_shape],
        filename=f"config_{config.object_shape}.json",
    )
    preflight_path = run_preflight_checks(config)
    append_log(config, f"Начат эксперимент для объекта {config.object_shape}; preflight={preflight_path}")

    project_root = resolve_project_root(config)
    metrics: List[Dict[str, float]] = []

    print("=" * 80)
    print(f"Объект: {config.object_shape}")
    print("=" * 80)

    print("Создание SAC-среды...")
    sac_env = make_vec_env(config)
    sac_model, sac_resumed, sac_resume_path = load_or_create_sac(config, sac_env)

    if sac_resumed:
        append_log(config, f"SAC продолжает обучение из checkpoint: {sac_resume_path}")
    else:
        append_log(config, "SAC начинает обучение с нуля")

    print("Обучение SAC...")
    start = time.perf_counter()
    sac_model.learn(
        total_timesteps=config.total_timesteps_sac,
        reset_num_timesteps=(
            config.reset_num_timesteps_on_resume
            if sac_resumed
            else True
        ),
        progress_bar=config.progress_bar,
    )
    sac_training_time = time.perf_counter() - start

    sac_path = _save_model_path(config, "SAC", resumed=sac_resumed)
    sac_model.save(str(sac_path))
    append_log(config, f"SAC обучен/дообучен за {sac_training_time:.3f} сек.; модель сохранена: {sac_path}.zip")

    print("Оценка SAC...")
    sac_metrics = evaluate_policy_custom(
        model=sac_model,
        config=config,
        algorithm_name="SAC",
        object_shape=config.object_shape,
    )
    sac_metrics["training_time_sec"] = sac_training_time
    metrics.append(sac_metrics)

    if config.save_gif:
        save_rollout_gif(
            model=sac_model,
            config=config,
            algorithm_name="SAC",
            object_shape=config.object_shape,
        )

    sac_env.close()

    print("Создание PPO-среды...")
    ppo_env = make_vec_env(config)
    ppo_model, ppo_resumed, ppo_resume_path = load_or_create_ppo(config, ppo_env)

    if ppo_resumed:
        append_log(config, f"PPO продолжает обучение из checkpoint: {ppo_resume_path}")
    else:
        append_log(config, "PPO начинает обучение с нуля")

    print("Обучение PPO...")
    start = time.perf_counter()
    ppo_model.learn(
        total_timesteps=config.total_timesteps_ppo,
        reset_num_timesteps=(
            config.reset_num_timesteps_on_resume
            if ppo_resumed
            else True
        ),
        progress_bar=config.progress_bar,
    )
    ppo_training_time = time.perf_counter() - start

    ppo_path = _save_model_path(config, "PPO", resumed=ppo_resumed)
    ppo_model.save(str(ppo_path))
    append_log(config, f"PPO обучен/дообучен за {ppo_training_time:.3f} сек.; модель сохранена: {ppo_path}.zip")

    print("Оценка PPO...")
    ppo_metrics = evaluate_policy_custom(
        model=ppo_model,
        config=config,
        algorithm_name="PPO",
        object_shape=config.object_shape,
    )
    ppo_metrics["training_time_sec"] = ppo_training_time
    metrics.append(ppo_metrics)

    if config.save_gif:
        save_rollout_gif(
            model=ppo_model,
            config=config,
            algorithm_name="PPO",
            object_shape=config.object_shape,
        )

    ppo_env.close()

    return metrics

def train_and_evaluate_multishape(
    config: ExperimentConfig,
    object_shapes: Iterable[str],
) -> List[Dict[str, float]]:
    """Запускает обучение и оценку для нескольких форм объекта."""

    all_metrics: List[Dict[str, float]] = []

    for object_shape in object_shapes:
        shape_config = ExperimentConfig(**asdict(config))
        shape_config.object_shape = object_shape

        shape_metrics = train_and_evaluate(shape_config)
        all_metrics.extend(shape_metrics)

    results_dir = resolve_project_root(config) / config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_metrics)
    output_path = results_dir / "sac_vs_ppo_metrics_ur10e_dg5f.csv"
    df.to_csv(output_path, index=False)

    append_log(config, f"Сохранена итоговая таблица метрик: {output_path}")
    save_config_snapshot(
        config,
        object_shapes=object_shapes,
        filename="last_experiment_config.json",
    )
    save_metrics_log(config, all_metrics)
    create_metric_figures(all_metrics, config)

    return all_metrics


def print_metrics(metrics: List[Dict[str, float]]) -> None:
    """Печатает метрики эксперимента в консоль."""

    for row in metrics:
        print("\n" + "=" * 80)
        print(f"Объект: {row['object']} | Алгоритм: {row['algorithm']}")
        for key, value in row.items():
            if key in {"object", "algorithm"}:
                continue
            if isinstance(value, (float, int)):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

# =============================================================================
# Блок 12. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="UR10e + DG-5F + ResNet18 + SAC/PPO experiment."
    )

    parser.add_argument(
        "--object-shapes",
        nargs="+",
        default=["cube"],
        choices=["cube", "cylinder", "sphere"],
        help="Формы объектов для обучения.",
    )

    parser.add_argument(
        "--timesteps-sac",
        type=int,
        default=10_000,
        help="Количество шагов обучения SAC.",
    )

    parser.add_argument(
        "--timesteps-ppo",
        type=int,
        default=10_000,
        help="Количество шагов обучения PPO.",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Количество эпизодов оценки.",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Максимальное число шагов в эпизоде.",
    )

    parser.add_argument(
        "--tesollo-root",
        type=str,
        default="tesollo_dg5f_mujoco-main",
        help="Путь к tesollo_dg5f_mujoco-main.",
    )

    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Не сохранять GIF rollout.",
    )

    return parser.parse_args()


def main() -> None:
    """Точка входа при запуске файла из командной строки."""

    args = parse_args()

    config = ExperimentConfig(
        project_root=str(Path(__file__).resolve().parent),
        tesollo_root=args.tesollo_root,
        total_timesteps_sac=args.timesteps_sac,
        total_timesteps_ppo=args.timesteps_ppo,
        eval_episodes=args.eval_episodes,
        max_episode_steps=args.max_episode_steps,
        save_gif=not args.no_gif,
    )

    metrics = train_and_evaluate_multishape(
        config=config,
        object_shapes=args.object_shapes,
    )

    print_metrics(metrics)


if __name__ == "__main__":
    main()