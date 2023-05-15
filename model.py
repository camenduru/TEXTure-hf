from __future__ import annotations

import datetime
import pathlib
import shlex
import subprocess
import sys
from typing import Generator, Optional

import gradio as gr

sys.path.append('TEXTurePaper')

from src.configs.train_config import GuideConfig, LogConfig, TrainConfig
from src.training.trainer import TEXTure


class Model:
    def __init__(self):
        self.max_num_faces = 100000

    def load_config(self, shape_path: str, text: str, seed: int,
                    guidance_scale: float) -> TrainConfig:
        text += ', {} view'

        log = LogConfig(exp_name=self.gen_exp_name())
        guide = GuideConfig(text=text)
        guide.background_img = 'TEXTurePaper/textures/brick_wall.png'
        guide.shape_path = 'TEXTurePaper/shapes/spot_triangulated.obj'
        config = TrainConfig(log=log, guide=guide)

        config.guide.shape_path = shape_path
        config.optim.seed = seed
        config.guide.guidance_scale = guidance_scale
        return config

    def gen_exp_name(self) -> str:
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d-%H-%M-%S')

    def check_num_faces(self, path: str) -> bool:
        with open(path) as f:
            lines = [line for line in f.readlines() if line.startswith('f')]
        return len(lines) <= self.max_num_faces

    def zip_results(self, exp_dir: pathlib.Path) -> str:
        mesh_dir = exp_dir / 'mesh'
        out_path = f'{exp_dir.name}.zip'
        subprocess.run(shlex.split(f'zip -r {out_path} {mesh_dir}'))
        return out_path

    def run(
        self, shape_path: str, text: str, seed: int, guidance_scale: float
    ) -> Generator[tuple[list[str], Optional[str], Optional[str], str], None,
                   None]:
        if not shape_path.endswith('.obj'):
            raise gr.Error('The input file is not .obj file.')
        if not self.check_num_faces(shape_path):
            raise gr.Error('The number of faces is over 100,000.')

        config = self.load_config(shape_path, text, seed, guidance_scale)
        trainer = TEXTure(config)

        trainer.mesh_model.train()

        total_steps = len(trainer.dataloaders['train'])
        for step, data in enumerate(trainer.dataloaders['train'], start=1):
            trainer.paint_step += 1
            trainer.paint_viewpoint(data)
            trainer.evaluate(trainer.dataloaders['val'],
                             trainer.eval_renders_path)
            trainer.mesh_model.train()

            sample_image_dir = config.log.exp_dir / 'vis' / 'eval'
            sample_image_paths = sorted(
                sample_image_dir.glob(f'step_{trainer.paint_step:05d}_*.jpg'))
            sample_image_paths = [
                path.as_posix() for path in sample_image_paths
            ]
            yield sample_image_paths, None, None, f'{step}/{total_steps}'

        trainer.mesh_model.change_default_to_median()
        trainer.full_eval()

        video_path = config.log.exp_dir / 'results' / 'step_00010_rgb.mp4'
        zip_path = self.zip_results(config.log.exp_dir)
        yield sample_image_paths, video_path.as_posix(), zip_path, 'Done!'
