#!/usr/bin/env python

from __future__ import annotations

import gradio as gr

from model import Model

DESCRIPTION = '''# TEXTure

This is an unofficial demo for [https://github.com/TEXTurePaper/TEXTurePaper](https://github.com/TEXTurePaper/TEXTurePaper).

This demo only accepts as input `.obj` files with less than 100,000 faces.

Inference takes about 10 minutes on a T4 GPU.
'''

model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            input_shape = gr.Model3D(label='Input 3D mesh')
            text = gr.Text(label='Text')
            seed = gr.Slider(label='Seed',
                             minimum=0,
                             maximum=100000,
                             value=3,
                             step=1)
            guidance_scale = gr.Slider(label='Guidance scale',
                                       minimum=0,
                                       maximum=50,
                                       value=7.5,
                                       step=0.1)
            run_button = gr.Button('Run')
        with gr.Column():
            result = gr.Video(label='Result')
            output_file = gr.File(label='Output mesh file')
    with gr.Row():
        examples = [
            ['shapes/dragon1.obj', 'a photo of a dragon', 0, 7.5],
            ['shapes/dragon2.obj', 'a photo of a dragon', 0, 7.5],
            ['shapes/eagle.obj', 'a photo of an eagle', 0, 7.5],
            ['shapes/napoleon.obj', 'a photo of Napoleon Bonaparte', 3, 7.5],
            ['shapes/nascar.obj', 'A next gen nascar', 2, 10],
        ]
        gr.Examples(examples=examples,
                    inputs=[
                        input_shape,
                        text,
                        seed,
                        guidance_scale,
                    ],
                    outputs=[
                        result,
                        output_file,
                    ],
                    cache_examples=False)

    run_button.click(fn=model.run,
                     inputs=[
                         input_shape,
                         text,
                         seed,
                         guidance_scale,
                     ],
                     outputs=[
                         result,
                         output_file,
                     ])

demo.queue(max_size=5).launch()
